#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin
"""Exports Fast-FoundationStereo (FFS) as the one ONNX graph FfsStereo builds
its TensorRT plan from: once per checkpoint and network size.

It runs NVlabs' own PyTorch code, from a local checkout of
https://github.com/NVlabs/Fast-FoundationStereo (commit 476f424 was used; its
code and research checkpoints are NVIDIA's, under their own licence, and this
package never vendors or fetches them), through the checkout's single-graph
export (scripts/make_single_onnx.py) with four changes that TensorRT needs to
build FFS fast and right on any size up to 1088 x 1920:

  1. the group-wise correlation volume built one disparity at a time, with
     each group's features normalised once: the same values, 25% faster;
  2. the GRU loop's grid_sample lookups cut into chunks of 32768 rows, and
     its 1 x 2 average pooling a reshape and a mean: the same values, but no
     launch past the 65535 TensorRT's GridSample takes (inputs past ~1 Mpx);
  3. channel attention's global max and mean pools as reductions: the same
     values, but no pooling window past TensorRT's kernel limit;
  4. the EdgeNeXt cross-covariance attention's normalised q k^T and softmax
     in FP32: TensorRT's fused kernel accumulates that long product badly in
     FP16.

The graph is traced under PyTorch's autocast (FP16), so it carries AMP's own
casts, and TensorRT (strongly typed) runs it in mixed precision as PyTorch
does. Its inputs are `left_image` and `right_image`, float 1 x 3 x H x W,
R'G'B' less ImageNet's mean over its deviation (FfsStereo's input kernel
makes them), and its output `disparity`, float 1 x 1 x H x W. H and W are
multiples of 32: a rig's size rounded up (960 x 544 stays; 1280 x 720 is
exported as 1280 x 736).

    venv/bin/python tools/ffs/export_onnx.py --ffs ~/Fast-FoundationStereo \\
        --size 960x544 --out ffs_23-36-37_960x544.onnx

It writes a JSON file beside the ONNX: checkpoint, size, iterations,
maximum disparity and the versions it ran with. Needs a CUDA GPU and the
checkout's Python requirements (torch, timm, omegaconf, onnx). Defaults: the
23-36-37 research checkpoint (chosen by measurement on a RealSense D455:
NVlabs' most accurate, at the commercial checkpoint's speed), 8 refinement
iterations, 192 pixels of maximum disparity.
"""

import argparse
import json
import os
import sys
import time

MEAN = [123.675, 116.28, 103.53]
DEVIATION = [58.395, 57.12, 57.375]


def gwc_volume_per_disparity(ref, tgt, maxdisp, num_groups, normalize=True):
    """The group-wise correlation volume (B, G, D, H, W), one disparity at a
    time. Normalising each group's channels per pixel commutes with the shift
    (the zero padding stays zero), so both maps are normalised once and only
    the (B, G, H, W) correlations are stacked, not D shifted copies of the
    C-channel features."""
    import torch
    import torch.nn.functional as F
    B, C, H, W = ref.shape
    cpg = C // num_groups
    if normalize:
        ref = F.normalize(ref.float().view(B, num_groups, cpg, H, W),
                          dim=2).view(B, C, H, W).to(ref.dtype)
        tgt = F.normalize(tgt.float().view(B, num_groups, cpg, H, W),
                          dim=2).view(B, C, H, W).to(tgt.dtype)
    outs = [(ref * F.pad(tgt, (d, 0, 0, 0))[:, :, :, :W])
            .view(B, num_groups, cpg, H, W).sum(2) for d in range(maxdisp)]
    return torch.stack(outs, 2).contiguous()


def chunked_bilinear_sampler(chunk):
    """core.utils.utils.bilinear_sampler with its grid_sample batch (a row a
    quarter-resolution pixel) cut into chunks of at most `chunk`: FFS's own
    low-memory path cuts at 102400, past the 65535 a TensorRT GridSample takes.
    The same values."""
    import torch
    import torch.nn.functional as F

    def bilinear_sampler(img, coords, mode='bilinear', mask=False,
                         low_memory=False, use1d=False):
        assert not mask
        W = img.shape[-1]
        coords = coords.clone()
        coords[..., 0] = 2 * coords[..., 0] / (W - 1) - 1
        return torch.cat([F.grid_sample(img[b:b + chunk], coords[b:b + chunk],
                                        align_corners=True)
                          for b in range(0, img.shape[0], chunk)], 0)
    return bilinear_sampler


class PairMeanF:
    """torch.nn.functional for core.geometry, with avg_pool2d(x, [1, 2],
    stride=[1, 2]) as a reshape and a mean: the same values, but no pooling
    launch over the (a row a quarter-resolution pixel) batch."""

    def __getattr__(self, name):
        import torch.nn.functional as F
        return getattr(F, name)

    @staticmethod
    def avg_pool2d(x, kernel_size, stride=None, **kw):
        import torch.nn.functional as F
        if list(kernel_size) == [1, 2] and list(stride) == [1, 2] and not kw:
            B, C, H, W = x.shape
            return x[..., :W - W % 2].reshape(B, C, H, W // 2, 2).mean(-1)
        return F.avg_pool2d(x, kernel_size, stride, **kw)


def fp32_xca():
    """timm's CrossCovarianceAttn.forward with normalize(q) normalize(k)^T and
    its softmax in FP32: the product reduces over every token (32k at 1080p),
    which TensorRT's fused attention accumulates badly in FP16."""
    import torch
    import torch.nn.functional as F

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 4, 1).unbind(0)
        with torch.autocast('cuda', enabled=False):
            attn = (F.normalize(q.float(), dim=-1)
                    @ F.normalize(k.float(), dim=-1).transpose(-2, -1))
            attn = (attn * self.temperature.float()).softmax(dim=-1)
        x = self.attn_drop(attn.to(v.dtype)) @ v
        return self.proj_drop(self.proj(x.permute(0, 3, 1, 2).reshape(B, N, C)))
    return forward


def global_pools():
    """AdaptiveMaxPool2d(1) and AdaptiveAvgPool2d(1) as modules that export as
    ReduceMax and ReduceMean: the same values."""
    import torch

    class GlobalMax(torch.nn.Module):
        def forward(self, x):
            return x.amax(dim=(2, 3), keepdim=True)

    class GlobalMean(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3), keepdim=True)
    return GlobalMax, GlobalMean


def parse_size(text):
    w, h = (int(v) for v in text.lower().split('x'))
    if w % 32 or h % 32:
        raise SystemExit(f'--size {text}: FFS needs multiples of 32 (round the '
                         'rig\'s size up)')
    return w, h


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ffs', required=True,
                    help='the NVlabs Fast-FoundationStereo checkout')
    ap.add_argument('--checkpoint',
                    help='model_best_bp2_serialize.pth (default: the '
                         'checkout\'s weights/23-36-37/)')
    ap.add_argument('--size', required=True,
                    help='WIDTHxHEIGHT, multiples of 32, e.g. 960x544')
    ap.add_argument('--iters', type=int, default=8,
                    help='GRU refinement iterations')
    ap.add_argument('--max_disp', type=int, default=192,
                    help='maximum disparity, pixels at full resolution')
    ap.add_argument('--out', required=True, help='the ONNX file to write')
    a = ap.parse_args()
    width, height = parse_size(a.size)
    ffs = os.path.abspath(os.path.expanduser(a.ffs))
    checkpoint = a.checkpoint or os.path.join(
        ffs, 'weights', '23-36-37', 'model_best_bp2_serialize.pth')

    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    sys.path.insert(0, ffs)
    sys.path.insert(0, os.path.join(ffs, 'scripts'))
    import torch
    from omegaconf import OmegaConf
    import make_single_onnx as single
    import core.geometry as geometry
    import timm.models.edgenext as edgenext

    torch.autograd.set_grad_enabled(False)
    model = torch.load(checkpoint, map_location='cpu', weights_only=False)
    OmegaConf.set_struct(model.args, False)
    if 'normalize' not in model.args:  # 15-44-51 pickles none; True is the
        model.args.normalize = True    # GWC kernels' default
    model.args.valid_iters = a.iters
    model.args.max_disp = a.max_disp
    model.args.mixed_precision = True
    model = model.cuda().eval()

    # The checkout's single-graph export, normalisation left to the caller.
    single._fs_module.normalize_image = lambda img: img
    single._fs_module.build_concat_volume_optimized_pytorch1 = \
        single._build_concat_volume_onnx
    # The four changes.
    single._fs_module.build_gwc_volume_optimized_pytorch1 = \
        gwc_volume_per_disparity
    geometry.bilinear_sampler = chunked_bilinear_sampler(32768)
    geometry.F = PairMeanF()
    GlobalMax, GlobalMean = global_pools()
    for module in model.modules():
        if type(module).__name__ == 'ChannelAttentionEnhancement':
            module.max_pool, module.avg_pool = GlobalMax(), GlobalMean()
    edgenext.CrossCovarianceAttn.forward = fp32_xca()

    wrapper = single.FastFoundationStereoSingleOnnx(model).cuda().eval()
    sample = (torch.randn(1, 3, height, width, device='cuda'),
              torch.randn(1, 3, height, width, device='cuda'))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    start = time.time()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        torch.onnx.export(wrapper, sample, a.out, opset_version=17,
                          input_names=['left_image', 'right_image'],
                          output_names=['disparity'],
                          do_constant_folding=True, dynamo=False)
    about = dict(checkpoint=os.path.abspath(checkpoint), width=width,
                 height=height, iters=a.iters, max_disp=a.max_disp,
                 input_mean=MEAN, input_deviation=DEVIATION,
                 changes=['gwc per disparity', 'grid_sample chunked 32768',
                          'global pools as reductions', 'xca fp32'],
                 precision='amp fp16 trace', torch=torch.__version__,
                 gpu=torch.cuda.get_device_name(0),
                 export_s=round(time.time() - start, 1))
    with open(os.path.splitext(a.out)[0] + '.json', 'w') as f:
        json.dump(about, f, indent=1)
    print(json.dumps(about))


if __name__ == '__main__':
    main()
