import numpy as np
import cv2
import datetime
from lxml import etree

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
import os
import argparse
import glob
import random

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def read_video(vid_path: str):
    vid = cv2.VideoCapture(vid_path)
    frames = []
    color_frames = []
    while True:
        ret, frame = vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            color_frames.append(frame_rgb)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)
        else:
            break
    vid.release()
    return np.array(frames), np.array(color_frames)

def save_img(img, output, save_path, cfg):
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    # [:, :, ::-1] converts from RBG to BGR and vice versa
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])