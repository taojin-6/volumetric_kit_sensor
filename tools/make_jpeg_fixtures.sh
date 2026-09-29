#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin
#
# Regenerates the committed JPEGs with the ffmpeg CLI: the first frame of the
# HEVC clips' patch pattern (tools/make_hevc_fixtures.sh), baseline JFIF, BT.601
# full range, at the highest quality, so a decoder's picture sits within a few
# codes of the pattern away from patch edges.
#
#   tests/data/jpeg/patches_256x144.jpg      4:2:0, what the Femto Mega sends
#   tests/data/jpeg/patches_255x143.jpg      4:2:0 at an odd size, whose chroma
#       planes round up
#   tests/data/jpeg/patches_422_256x144.jpg  4:2:2, which GpuFramePrep cannot
#       read, so a decoder converts it to 4:2:0 on the host
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/tests/data/jpeg"
mkdir -p "$out"

# The HEVC clips' frame 0: patch p = (column + 3 * row) mod 8 over 32x72 luma
# patches; Y = 40 + 24p, U = 64 + 16 (3p mod 8), V = 64 + 16 (5p mod 8).
p_luma='mod(floor(X/32)+3*floor(Y/72),8)'
p_chroma_420='mod(floor(X/16)+3*floor(Y/36),8)'
p_chroma_422='mod(floor(X/16)+3*floor(Y/72),8)'
# Tagged full range, not converted to it: FFmpeg's JPEG encoder takes only full
# range, and would stretch a limited-range pattern to reach it.
pattern() {  # size, pixel format, chroma patch expression
  echo "nullsrc=s=$1:r=1,format=$2,geq=lum='40+24*${p_luma}':cb='64+16*mod(3*$3,8)':cr='64+16*mod(5*$3,8)',setparams=range=pc"
}
jpeg() {  # size, pixel format, chroma expression, file
  ffmpeg -hide_banner -loglevel error -y -f lavfi -i "$(pattern "$1" "$2" "$3")" \
    -frames:v 1 -c:v mjpeg -q:v 1 -qmin 1 -f image2 "$out/$4"
}
jpeg 256x144 yuv420p "$p_chroma_420" patches_256x144.jpg
jpeg 255x143 yuv420p "$p_chroma_420" patches_255x143.jpg
jpeg 256x144 yuv422p "$p_chroma_422" patches_422_256x144.jpg
