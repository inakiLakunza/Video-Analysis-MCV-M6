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


def make_video(out_folder="./", img_folder=None, name="example_video", start=800, end=1000):
    """
    Make a .mp4 from the estimation
    https://stackoverflow.com/questions/62880911/generate-video-from-numpy-arrays-with-opencv

    Parameters
        estimation : np.ndarray([1606, 1080, 1920, 3], dtype=uint8)
    """

    if img_folder is None:
        img_folder = out_folder
    
    duration = 2141
    img_shape = 1920, 1080 
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video=cv2.VideoWriter(os.path.join(out_folder, name+".avi"), fourcc, fps, (img_shape[0], img_shape[1]), True)

    
    for j in range(start, end):
        img = cv2.imread(os.path.join(img_folder, "frame_"+str(j)+".png"))
        video.write(img)


    