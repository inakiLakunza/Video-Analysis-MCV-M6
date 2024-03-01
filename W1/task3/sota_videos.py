
import numpy as np
import cv2
from cv2.bgsegm import createBackgroundSubtractorCNT
from cv2.bgsegm import createBackgroundSubtractorGMG
from cv2.bgsegm import createBackgroundSubtractorGSOC
from cv2.bgsegm import createBackgroundSubtractorLSBP
from cv2.bgsegm import createBackgroundSubtractorMOG

import datetime
import matplotlib.pyplot as plt
import rembg
from PIL import Image
from utils import *
from tqdm import tqdm
import pickle
import os
import argparse
import glob
import time


# NUMBER OF FRAMES IN THE FIRST 25%
N_TRAIN_FRAMES = 535

# NAMES OF THE USED STATES OF THE ART
MODEL_NAMES = {
    0: "MOG",
    1: "MOG2",
    2: "GMG",
    3: "KNN",
    4: "LSBP",
    5: "CNT",
    6: "GSOC",
    7: "REMBG"
}



def videos_sota(vid_path):

    gray_frames, color_frames = read_video(vid_path)

    substractor1 = cv2.bgsegm.createBackgroundSubtractorMOG()   
    substractor2 = cv2.createBackgroundSubtractorMOG2()
    substractor3 = cv2.bgsegm.createBackgroundSubtractorGMG()
    substractor4 = cv2.createBackgroundSubtractorKNN() 
    substractor5 = cv2.bgsegm.createBackgroundSubtractorLSBP()
    substractor6 = cv2.bgsegm.createBackgroundSubtractorCNT()

    substractors = [substractor1, substractor2, substractor3,
                    substractor4, substractor5, substractor6]

    # Split 25-75 frames
    gray_frames_25, gray_frames_75 = split_frames(gray_frames)
    color_frames_25, color_frames_75 = split_frames(color_frames)


    for sub_i in range(len(substractors)):
        subs = substractors[sub_i]

        # Train  the substractor with the first 25% frames
        for frame in gray_frames_25: subs.apply(frame)

        estimation = estimate_sota_foreground(subs, gray_frames_75, sub_i)

        video_name = "./sota_videos/vid_"+MODEL_NAMES[sub_i]+".mp4"
        print(estimation.shape)
        print(estimation.dtype)
        make_video(estimation, video_name)


# Estimate state of the art foregrounds
def estimate_sota_foreground(subs, frames, model_index):

    estimation = []
    for i in range(len(frames)):
        frame = frames[i]
        # Apply model, and do not learn from it
        single_est = subs.apply(frame, learningRate=0.0)

        if i==1200:
            img_name = "./plotting_results/"+"frame1200_"+MODEL_NAMES[model_index]+".jpg"
            plt.imsave(img_name, single_est, cmap="gray")

        # there are some models which return more than binary
        if len(single_est.shape)==3:
            single_est = cv2.cvtColor(single_est, cv2.COLOR_RGB2GRAY)
            
            _, single_est = cv2.threshold(single_est, 200, 255, cv2.THRESH_BINARY )
    
        boolean_est = single_est.astype(bool)
        estimation.append(boolean_est)

    return np.array(estimation)




description = """
INDEXES OF THE SUBSTRACTORS:
    \n 0: MOG
    \n 1: MOG2
    \n 2: GMG
    \n 3: KNN
    \n 4: LSBP
    \n 5: CNT
    \n 6: GSOC
    \n 7: rembg (special)
    """

vid_path = '/ghome/group07/test/AICity_data/train/S03/c010/vdo.avi'

videos_sota(vid_path)