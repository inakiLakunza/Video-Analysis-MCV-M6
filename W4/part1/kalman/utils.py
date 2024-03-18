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


def make_video(estimation):
    """
    Make a .mp4 from the estimation
    https://stackoverflow.com/questions/62880911/generate-video-from-numpy-arrays-with-opencv

    Parameters
        estimation : np.ndarray([1606, 1080, 1920, 3], dtype=uint8)
    """
    size = estimation.shape[1], estimation.shape[2]
    duration = estimation.shape[0]
    fps = 10
    os.makedirs('./vid/', exist_ok=True)
    out = cv2.VideoWriter(f'./vid/estimation_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    for i in range(duration):
        data = (estimation[i]).astype(np.uint8)
        # I am converting the data to gray but we should look into this...
        # data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        out.write(data)
    out.release()
    print("Video done.")


    