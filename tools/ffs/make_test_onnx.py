#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin
"""Writes the FFS test models in tests/data/ffs: FFS's inputs and output, and a
disparity a test can work out by hand.

Each takes `left_image` and `right_image`, float 1 x 3 x 32 x 64, normalised as
FFS's are ((value - mean) / deviation, ImageNet's), and gives `disparity`,
float 1 x 1 x 32 x 64: the normalisation undone, then a weighted sum over the
channels of left less right,

    disparity = sum_c weight[c] * (left[c] - right[c])

with weights (1/8, 2/8, 5/8) in difference_64x32.onnx and half of them in
half_difference_64x32.onnx. So a grey pair's disparity is left less right
(or half of it), and a coloured one's tells the channels apart. Opset 17, as
tools/ffs/export_onnx.py writes FFS.

    python3 -m venv venv && venv/bin/pip install onnx numpy
    venv/bin/python tools/ffs/make_test_onnx.py tests/data/ffs

The output is the same bytes each run.
"""

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MEAN = [123.675, 116.28, 103.53]
DEVIATION = [58.395, 57.12, 57.375]
HEIGHT, WIDTH = 32, 64


def const(name, values, shape, dtype=np.float32):
    return numpy_helper.from_array(
        np.array(values, dtype=dtype).reshape(shape), name)


def model(weights):
    shape = [1, 3, HEIGHT, WIDTH]
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape)
              for n in ('left_image', 'right_image')]
    output = helper.make_tensor_value_info('disparity', TensorProto.FLOAT,
                                           [1, 1, HEIGHT, WIDTH])
    inits = [const('deviation', DEVIATION, [1, 3, 1, 1]),
             const('mean', MEAN, [1, 3, 1, 1]),
             const('weights', weights, [1, 3, 1, 1]),
             const('channels', [1], [1], np.int64)]
    nodes = []
    for side in ('left', 'right'):
        nodes.append(helper.make_node('Mul', [f'{side}_image', 'deviation'],
                                      [f'{side}_scaled']))
        nodes.append(helper.make_node('Add', [f'{side}_scaled', 'mean'],
                                      [f'{side}_value']))
    nodes.append(helper.make_node('Sub', ['left_value', 'right_value'],
                                  ['difference']))
    nodes.append(helper.make_node('Mul', ['difference', 'weights'],
                                  ['weighted']))
    nodes.append(helper.make_node('ReduceSum', ['weighted', 'channels'],
                                  ['disparity'], keepdims=1))
    graph = helper.make_graph(nodes, 'ffs_test', inputs, [output], inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)],
                          producer_name='make_test_onnx.py')
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', help='the directory to write the models in')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, weights in (('difference', [0.125, 0.25, 0.625]),
                          ('half_difference', [0.0625, 0.125, 0.3125])):
        path = os.path.join(a.out, f'{name}_{WIDTH}x{HEIGHT}.onnx')
        onnx.save(model(weights), path)
        print(path, os.path.getsize(path), 'bytes')


if __name__ == '__main__':
    main()
