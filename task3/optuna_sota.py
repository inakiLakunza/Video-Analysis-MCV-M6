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
import optuna


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


def state_of_the_art(vid_path, model, model_i):

    gray_frames, color_frames = read_video(vid_path)

    gt = read_annotations(annotations_path)

    # Split 25-75 frames
    gray_frames_25, gray_frames_75 = split_frames(gray_frames)
    color_frames_25, color_frames_75 = split_frames(color_frames)

    start_time = time.time()

    # Train  the substractor with the first 25% frames
    for frame in gray_frames_25: model.apply(frame)

    estimation = estimate_sota_foreground(model, gray_frames_75, model_i)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The time needed by the model {MODEL_NAMES[model_i]} has been: {elapsed_time} seconds")

    model_times.append(elapsed_time)

    # Separate objects and compute metrics
    precision_list = []
    recall_list = []
    for frame_idx in (pbar := tqdm(range(estimation.shape[0]))):
    #for frame_idx in (range(estimation.shape[0])):
        precision, recall = connected_components(frame_idx, color_frames_25.shape[0], estimation[frame_idx], color_frames_75[frame_idx], gt)
        precision_list.append(precision)
        recall_list.append(recall)
        pbar.set_description(f"[Model name: {MODEL_NAMES[model_i]}] Current AP {round(sum(precision_list) / len(precision_list), 2)}")


    AP = sum(precision_list) / len(precision_list)
    AR = sum(recall_list) / len(recall_list)

    return AP, AR

        


# Estimate state of the art foregrounds
def estimate_sota_foreground(subs, frames, model_index):

    estimation = []
    for i in range(len(frames)):
        frame = frames[i]
        # Apply model, and do not learn from it
        single_est = subs.apply(frame, learningRate=0.0)

        # there are some models which return more than binary
        if len(single_est.shape)==3:
            single_est = cv2.cvtColor(single_est, cv2.COLOR_RGB2GRAY)
            
            _, single_est = cv2.threshold(single_est, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        boolean_est = single_est.astype(bool)
        estimation.append(boolean_est)

    return np.array(estimation)


def objective(trial):

    model=None

    if s_index == 0:   
        model = cv2.bgsegm.createBackgroundSubtractorMOG(
            nmixtures=trial.suggest_int("nmixtures", 2, 8) 
        )

    if s_index == 2:   
        model = cv2.bgsegm.createBackgroundSubtractorGMG(
             initializationFrames=trial.suggest_categorical("initializationFrames", [120, 250, 535]),
             decisionThreshold=trial.suggest_float("decisionThreshold", 0.4, 1.0)
        )

    if s_index == 4:   
        model = cv2.bgsegm.createBackgroundSubtractorLSBP(
            nSamples=trial.suggest_int("nSamples", 15, 25),
            LSBPRadius=trial.suggest_int("LSBPRadius", 12, 20),
            minCount=trial.suggest_int("minCount", 1, 5) 
        )

    if s_index == 6:   
        model = cv2.bgsegm.createBackgroundSubtractorCNT(
            minPixelStability=trial.suggest_int("minPixelStability", 10, 20)
        )

    AP, AR = state_of_the_art(vid_path, model, s_index)

    return AP
        



if __name__ == "__main__":

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

    model_times = []

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--index', type=int, help=description, default=0)


    args = parser.parse_args()

    s_index = args.index

    vid_path = '/ghome/group07/test/AICity_data/train/S03/c010/vdo.avi'
    annotations_path = '/ghome/group07/test/ai_challenge_s03_c010-full_annotation.xml'

    if s_index == 0:
        search_space = {
            "nmixtures": [2, 3, 4, 5, 6, 7, 8]
        }

    if s_index == 2:
        search_space = {
            "initializationFrames": [120, 250, 535],
            "decisionThreshold": [0.4, 1.0]
        }

    if s_index == 4:
        search_space = {
            "nSamples": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            "LSBPRadius": [12, 13, 14, 15, 16, 17, 18, 19, 20],
            "minCount": [1, 2, 3, 4, 5]
        }

    if s_index == 6:
        search_space = {
            "minPixelStability": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        }

    storage_name = "sqlite:///{}.db".format(MODEL_NAMES[s_index])
    study_name = "study-{}".format(MODEL_NAMES[s_index])
    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space),
        direction="maximize",
        storage=storage_name,
        study_name=study_name,
    )
    if s_index in [4]: study.optimize(objective, n_trials=20)
    else: study.optimize(objective)

    time_pickle = './model_logs_optuna/times_'+ MODEL_NAMES[s_index]+'.pkl'
    with open(time_pickle, 'wb') as f:
        pickle.dump(model_times, f)

    