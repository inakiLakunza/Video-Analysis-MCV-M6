# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2


def compute_flow_pipeline(im1_path: str, im2_path: str, channels: int = 3, **kwargs):
    # read images
    im1 = cv2.imread(im1_path, cv2.COLOR_BGR2RGB)
    im2 = cv2.imread(im2_path, cv2.COLOR_BGR2RGB)
    if channels == 1:
        im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
        im1 = im1.reshape([im1.shape[0], im1.shape[1], 1])
        im2 = im2.reshape([im2.shape[0], im2.shape[1], 1])
    
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # params
    alpha = float(kwargs['alpha']) # 0.012
    ratio = float(kwargs['ratio']) # 0.75
    minWidth = int(kwargs['minWidth']) # 20
    nOuterFPIterations = int(kwargs['nOuterFPIterations']) #7
    nInnerFPIterations = int(kwargs['nInnerFPIterations']) #1
    nSORIterations = int(kwargs['nSORIterations']) #30
    colType = 0 if channels > 1 else 1

    # compute flow
    u, v, _ = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    return np.dstack((u, v))


def visualize_flow(im1_path: str, flow):
    im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
    im1 = im1.reshape([im1.shape[0], im1.shape[1], 1])
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb
