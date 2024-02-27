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
import utils


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


def state_of_the_art(vid_path, ind):

    print(f"The chosen state of the art model is: {MODEL_NAMES[ind]}")

    gray_frames, color_frames = utils.read_video(vid_path)
    #print(f"gray_frames.shape: {gray_frames.shape} \t color_frames.shape: {color_frames.shape}")

    gt = utils.read_annotations(annotations_path)

    # Split 25-75 frames
    gray_frames_25, gray_frames_75 = utils.split_frames(gray_frames)
    color_frames_25, color_frames_75 = utils.split_frames(color_frames)

    # Select substractor model
    if ind==0:   subs = createBackgroundSubtractorMOG()
    elif ind==1: subs = cv2.createBackgroundSubtractorMOG2()
    elif ind==2: subs = createBackgroundSubtractorGMG()
    elif ind==3: subs = cv2.createBackgroundSubtractorKNN()
    elif ind==4: subs = createBackgroundSubtractorLSBP()
    elif ind==5: subs = createBackgroundSubtractorCNT()
    elif ind==6: subs = createBackgroundSubtractorGSOC()

    
    elif ind==7:
        # SAVE ALL ESTIMATIONS OF THE REMBG OUTPUT - DONE
        #-------------------------------------------------
        #for i in range(len(gray_frames)):
        #    frame = gray_frames[i]
        #    print(f"Working with frame {i} out of {len(gray_frames)} frames")
        #    estimation.append(use_rembg(frame, i))
        #-------------------------------------------------------------

        estimation = use_rembg_outs_for_estimation(rembg_outs)
        print("All estimations with the U-Net based model completed")
        
    if ind in range(7):
        # Train  the substractor with the first 25% frames
        for frame in gray_frames_25: subs.apply(frame)

        estimation = estimate_sota_foreground(subs, gray_frames_75, ind)

    # Separate objects and compute metrics
    precision_list = []
    recall_list = []
    #for frame_idx in (pbar := tqdm(range(estimation.shape[0]))):
    for frame_idx in (range(estimation.shape[0])):
        print(f"Computing precision and recall with frame {frame_idx}")
        precision, recall = connected_components(frame_idx, color_frames_25.shape[0], estimation[frame_idx], color_frames_75[frame_idx], gt)
        precision_list.append(precision)
        recall_list.append(recall)
        pbar.set_description(f"[Model name: {MODEL_NAMES[ind]}] Current AP {round(sum(precision_list) / len(precision_list), 2)}")


    AP = sum(precision_list) / len(precision_list)
    AR = sum(recall_list) / len(recall_list)

    print(f"[Model name: {MODEL_NAMES[ind]}] Average Preicison = {AP}")
    print(f"[Model name: {MODEL_NAMES[ind]}] Average Recall = {AR}")

    # Save as pkl so that we can later make plots
    log_AP_name = './model_logs/AP_model_'+str(MODEL_NAMES[ind])+'.pkl'
    with open(log_AP_name, 'wb') as file:
        pickle.dump({'precision_list': precision_list}, file)
    
    log_AR_name = './model_logs/AR_model_'+str(MODEL_NAMES[ind])+'.pkl'
    with open(log_AR_name, 'wb') as file:
        pickle.dump({'recall_list': recall_list}, file)


    

# Estimate state of the art foregrounds
def estimate_sota_foreground(subs, frames, model_index):

    estimation = []
    for frame in frames:
        # Apply model, and do not learn from it
        single_est = subs.apply(frame, learningRate=0.0)

        # there are some models which return more than binary
        if len(single_est.shape)==3:
            single_est = cv2.cvtColor(single_est, cv2.COLOR_RGB2GRAY)
            
            _, single_est = cv2.threshold(single_est, 200, 255, cv2.THRESH_BINARY)
    
        boolean_est = single_est.astype(bool)
        estimation.append(boolean_est)

    return np.array(estimation)
    

def use_rembg_outs_for_estimation(path):
    
    estimations = []
    for i in range(2141):
        img_path = path+"/frame_"+str(i)+".jpg"
        print(f"Analyzing frame: {i}")
        im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        height, width = im.shape

        #single_estimation = [[False] * width for _ in range(height)]

        #for y in range(height):
        #    for x in range(width):
        #        if im[y][x] != 0:
        #            single_estimation[y][x] = True

        single_estimation = [[im[y][x] != 0 for x in range(width)] for y in range(height)]


        estimations.append(single_estimation)
    return np.array(estimations)


def create_rembg_outs(frame, i):
    # Convert the input image to a numpy array
    input_array = np.array(frame)

    # Apply background removal using rembg
    output_array = rembg.remove(input_array, session=rembg_session, alpha_matting=True, alpha_matting_foreground_threshold=270,alpha_matting_background_threshold=20, alpha_matting_erode_size=11)

    # SAVE OUTPUTS OF THE MODEL
    #--------------------------------
    # Create a PIL Image from the output array
    output_image = Image.fromarray(output_array)

    rgb_im = output_image.convert('RGB')

    # Save the output image
    save_path = rembg_output_path + "frame_" + str(i) + ".jpg"
    rgb_im.save(save_path)
    #------------------------------------------




    '''
    # Convert the input image to a numpy array
    input_array = np.array(frames[1200])

    # Apply background removal using rembg
    output_array = rembg.remove(input_array)

    # Create a PIL Image from the output array
    output_image = Image.fromarray(output_array)

    rgb_im = output_image.convert('RGB')

    # Save the output image
    rgb_im.save('output_image.jpg')
    '''


if __name__ == "__main__":

    description = """´
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

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--index', type=int, help=description, default=0)


    args = parser.parse_args()

    s_index = args.index
    if s_index==7:
        # REMBG OUTPUTS SAVED ALREADY
        #model_name = "unet"
        #rembg_session = rembg.new_session(model_name)
        rembg_outs = '/ghome/group07/test/task3/rembg_outputs'

    vid_path = '/ghome/group07/test/AICity_data/train/S03/c010/vdo.avi'
    annotations_path = '/ghome/group07/test/ai_challenge_s03_c010-full_annotation.xml'
    rembg_output_path = "/ghome/group07/test/task3/rembg_outputs/"
    state_of_the_art(vid_path, s_index)