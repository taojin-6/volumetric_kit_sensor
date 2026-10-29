#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin
"""OpenCV's stereoRectify on the pairs tests/sensor_stereo_rectifier_test.cpp
pins, printed as that test's kOpenCvCases initializer.

The build never needs OpenCV: these numbers are generated once and committed
in the test. Regenerate them, after changing a case here, with OpenCV in a
virtual environment of its own:

    python3 -m venv venv && venv/bin/pip install opencv-python-headless numpy
    venv/bin/python tools/stereo_rectify_reference.py

and paste the output over the test's kOpenCvCases. The flags are OpenCV's
default, CALIB_ZERO_DISPARITY, which StereoRectifier keeps: both views share
one principal point.
"""

import cv2
import numpy as np

# Each pair: the two cameras (size, fx, fy, cx, cy, and OpenCV's eight rational
# coefficients k1 k2 p1 p2 k3 k4 k5 k6), the left camera's point in the right
# camera as OpenCV's R (a Rodrigues vector) and T (metres), the rectified size
# (0 x 0: the cameras') and alpha.
MILD_LEFT = ((1280, 720), (652.5, 651.75, 641.25, 358.5),
             (-0.052, 0.031, 0.0007, -0.0004, -0.0061, 0.012, 0.0021,
              -0.0009))
MILD_RIGHT = ((1280, 720), (649.25, 650.5, 636.75, 362.25),
              (-0.047, 0.026, -0.0003, 0.0006, -0.0048, 0.009, 0.0, 0.0))
MILD_R = (0.0123, -0.0347, 0.0061)
MILD_T = (-0.09503, 0.0012, -0.0021)

STRONG_LEFT = ((800, 600), (420.0, 421.0, 401.0, 297.0),
               (-0.28, 0.09, 0.0012, -0.0009, -0.01, 0.05, 0.0, 0.0))
STRONG_RIGHT = ((800, 600), (418.5, 419.25, 396.5, 303.75),
                (-0.26, 0.08, -0.0006, 0.0011, -0.008, 0.04, 0.0, 0.0))
STRONG_R = (0.02, 0.11, -0.015)
STRONG_T = (-0.12, 0.004, 0.009)

CASES = [
    ("mild lenses, toed in, to 960x544, alpha 0", MILD_LEFT, MILD_RIGHT,
     MILD_R, MILD_T, (960, 544), 0.0),
    ("mild lenses, toed in, to 960x544, alpha 1", MILD_LEFT, MILD_RIGHT,
     MILD_R, MILD_T, (960, 544), 1.0),
    ("mild lenses, toed in, own size, alpha 0.5", MILD_LEFT, MILD_RIGHT,
     MILD_R, MILD_T, (0, 0), 0.5),
    ("strong barrel lenses, converging, to 640x480, alpha 0", STRONG_LEFT,
     STRONG_RIGHT, STRONG_R, STRONG_T, (640, 480), 0.0),
    ("strong barrel lenses, converging, to 640x480, alpha 1", STRONG_LEFT,
     STRONG_RIGHT, STRONG_R, STRONG_T, (640, 480), 1.0),
]


def num(v):
    return repr(float(v))


def camera(c):
    (w, h), (fx, fy, cx, cy), d = c
    return ("{{%d, %d}, {%s, %s, %s, %s}, {%s}}" %
            (w, h, num(fx), num(fy), num(cx), num(cy), ", ".join(
                num(x) for x in d)))


def matrix(m):
    return "{" + ", ".join(num(x) for x in np.asarray(m).ravel()) + "}"


def main():
    print("// tools/stereo_rectify_reference.py, OpenCV %s" % cv2.__version__)
    for name, left, right, rvec, tvec, size, alpha in CASES:
        k = []
        d = []
        for c in (left, right):
            _, (fx, fy, cx, cy), dist = c
            k.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                              dtype=np.float64))
            d.append(np.array(dist, dtype=np.float64))
        rotation, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
        r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
            k[0], d[0], k[1], d[1], left[0], rotation,
            np.array(tvec, dtype=np.float64).reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=alpha, newImageSize=size)
        print("    {\"%s\"," % name)
        print("     %s," % camera(left))
        print("     %s," % camera(right))
        print("     {%s}," % ", ".join(num(x) for x in rvec))
        print("     {%s}," % ", ".join(num(x) for x in tvec))
        print("     {%d, %d}," % size)
        print("     %s," % num(alpha))
        print("     %s," % matrix(r1))
        print("     %s," % matrix(r2))
        print("     %s," % matrix(p1))
        print("     %s}," % matrix(p2))


if __name__ == "__main__":
    main()
