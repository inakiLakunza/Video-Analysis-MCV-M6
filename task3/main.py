import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import rembg
from PIL import Image
import utils
from tqdm import tqdm
import pickle
import os
import argparse

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
    6: "REMBG"
}

def read_video(vid_path: str):
    vid = cv2.VideoCapture(vid_path)
    frames = []
    while True:
        ret, frame = vid.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)
        else:
            break
    vid.release()
    return np.array(frames)


def estimate_foreground_sequential(frames, mean_, std_, alpha_ = 2):
    """
    I did this sequentially xd DO NOT USE THIS, use the vectorized version 
    """
    frames = frames[int(frames.shape[0] * 0.25):, :, :]
    res = np.empty_like(frames, dtype=bool)

    frames_, rows, cols, channels = frames.shape
    for f in range(frames_):
        for c in range(channels):
            for i in range(rows):
                for j in range(cols):
                    pixel = frames[f, i, j, c] 
                    if abs(pixel - mean_[i, j, c]) >= alpha_ * (std_[i, j, c] + 2):
                        res[f, i, j, c] = 1
                    else:
                        res[f, i, j, c] = 0
    return res


def estimate_foreground(frames, mean_, std_, alpha_ = 2):
    """
    Compute estimation for the other 75% part of the video

    Parameters
        frames : np.ndarray([2141, 1080, 1920, 3])
        mean : np.ndarray([1080, 1920, 3])
        std : np.ndarray([1080, 1920, 3])
        alpha : int parameter to decide
    
    Returns
        condition : np.ndarray([1606, 1080, 1920, 3], dtype=bool)
    """
    frames = frames[int(frames.shape[0] * 0.25):, :, :]
    estimation = np.abs(frames - mean_) >= (alpha_ * (std_ + 2))
    return estimation.astype(bool)


def make_video(estimation):
    """
    Make a .mp4 from the estimation
    https://stackoverflow.com/questions/62880911/generate-video-from-numpy-arrays-with-opencv

    Parameters
        estimation : np.ndarray([1606, 1080, 1920, 3], dtype=bool)
    """
    size = estimation.shape[1], estimation.shape[2]
    duration = estimation.shape[0]
    fps = 10
    out = cv2.VideoWriter(f'./estimation_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for i in range(duration):
        data = (estimation[i] * 255).astype(np.uint8)
        # I am converting the data to gray but we should look into this...
        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        out.write(data)
    out.release()

def connected_components(frame_idx, inc, gray_frame, color_frame, gt):
    """
    Separate into objects.

    He hecho que el ground truth sea un diccionario de frames,
    y dentro de cada frames hay los bounding boxes.
    (es una fumada el .xml q nos dan .......)
    Por ejemplo:

        frame["86"] = [
            {xtl: _, ytl: _, xbr: _, ybr: _},
            {xtl: _, ytl: _, xbr: _, ybr: _}
        ]

    Parameters:
        frame_idx : int
        inc : int   increment value because we are working with the last 75%, 
                    our frame 0 is not the first frame of the video
        gray_frame : np.ndarray([1080, 1920], dtype=bool)
        color_frame : np.ndarray([1080, 1920, 3], dtype=uint8)
    """
    gray_frame = (gray_frame * 255).astype(np.uint8)

    #plt.imsave(f'./pruebas_1_1/before_{frame_idx + inc}.jpg', gray_frame, cmap='gray')

    # Opening
    # kernel = np.ones((3,3),np.uint8)
    # gray_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_OPEN, kernel)

    # Connected components
    analysis = cv2.connectedComponentsWithStats(gray_frame, 4, cv2.CV_32S) 
    (totalLabels, label_ids, values, centroid) = analysis 
    output = np.zeros(gray_frame.shape, dtype="uint8")

    # Create mask1 and mask2 to compute metrics
    pred_mask_list = []

    # Loop through each component 
    for i in range(1, totalLabels): 
        area = values[i, cv2.CC_STAT_AREA] 
        pred_mask = np.zeros(gray_frame.shape, dtype="uint8") # prediction

        if (area > 5_000) and (area < 250_000): 
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)

            # Bounding box
            x = values[i, cv2.CC_STAT_LEFT]
            y = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]
            area = values[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroid[i]
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.rectangle(pred_mask, (x, y), (x + w, y + h), 255, -1)

            #plt.imsave(f'./pruebas_1_1/pred_mask_{len(pred_mask_list)}.jpg', pred_mask, cmap="gray")

            pred_mask_list.append(pred_mask)

    # Paint the GT Bounding boxes
    gt_mask_list = []

    real_frame_idx = str(frame_idx + inc)
    if real_frame_idx in gt:
        for box in gt[real_frame_idx]:
            gt_mask = np.zeros(gray_frame.shape, dtype="uint8") # gt
            xtl = int(float(box['xtl']))
            ytl = int(float(box['ytl']))
            xbr = int(float(box['xbr']))
            ybr = int(float(box['ybr']))
            
            cv2.rectangle(color_frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 3)
            cv2.rectangle(gt_mask, (xtl, ytl), (xbr, ybr), 255, -1)

            #plt.imsave(f'./pruebas_1_1/gt_mask_{len(gt_mask_list)}.jpg', gt_mask, cmap="gray")

            gt_mask_list.append(gt_mask)
    
    # Compute metrics
    precision = utils.compute_metric(pred_mask_list, gt_mask_list, threshold=0.5)
    recall = utils.compute_metric(gt_mask_list, pred_mask_list, threshold=0.5)

    # plt.imsave(f'./pruebas_1_1/after_pred_mask_{frame_idx + inc}.jpg', pred_mask, cmap="gray")
    # plt.imsave(f'./pruebas_1_1/after_gt_mask_{frame_idx + inc}.jpg', gt_mask, cmap="gray")

    #plt.imsave(f'./pruebas_1_1/after_{frame_idx + inc}.jpg', color_frame)

    return precision, recall


def state_of_the_art(vid_path, ind):

    print(f"The chosen state of the art model is: {MODEL_NAMES[ind]}")

    gray_frames, color_frames = utils.read_video(vid_path)
    #print(f"gray_frames.shape: {gray_frames.shape} \t color_frames.shape: {color_frames.shape}")

    gt = utils.read_annotations(annotations_path)

    # Split 25-75 frames
    gray_frames_25, gray_frames_75 = utils.split_frames(gray_frames)
    color_frames_25, color_frames_75 = utils.split_frames(color_frames)

    # Select substractor model
    if ind==0:   subs = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif ind==1: subs = cv2.createBackgroundSubtractorMOG2()
    elif ind==2: subs = cv2.bgsegm.createBackgroundSubtractorGMG()
    elif ind==3: subs = cv2.createBackgroundSubtractorKNN()
    elif ind==4: subs = cv2.bgsegm.createBackgroundSubtractorLSBP()
    elif ind==5: subs = cv2.bgsegm.createBackgroundSubtractorCNT()
    
    elif ind==6:
        use_rembg()
        return 0
    
    # Train  the substractor with the first 25% frames
    for frame in gray_frames_25: subs.apply(frame)

    estimation = estimate_sota_foreground(subs, gray_frames_75, ind)

    # Separate objects and compute metrics
    precision_list = []
    recall_list = []
    for frame_idx in (pbar := tqdm(range(estimation.shape[0]))):
        precision, recall = connected_components(frame_idx, color_frames_25.shape[0], estimation[frame_idx], color_frames_75[frame_idx], gt)
        precision_list.append(precision)
        recall_list.append(recall)
        pbar.set_description(f"[Model name: {MODEL_NAMES[ind]}] Current AP {round(sum(precision_list) / len(precision_list), 2)}")


    AP = sum(precision_list) / len(precision_list)
    AR = sum(recall_list) / len(recall_list)

    print(f"[Model name: {MODEL_NAMES[ind]}] Average Preicison = {AP}")
    print(f"[Model name: {MODEL_NAMES[ind]}] Average Recall = {AR}")

    # Save as pkl so that we can later make plots
    log_AP_name = './model_ltogs/AP_model_{MODEL_NAMES[ind]}.pkl'
    with open(log_AP_name, 'wb') as file:
        pickle.dump({'precision_list': precision_list}, file)
    
    log_AR_name = './model_logs/AR_model_{MODEL_NAMES[ind]}.pkl'
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
            
            _, single_est = cv2.threshold(single_est, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        boolean_est = single_est.astype(bool)
        estimation.append(boolean_est)

    return np.array(estimation)
    


def use_rembg():
    # to complete
    pass


def try_state_of_the_art(vid_path):
    frames = read_video(vid_path)
    print(f"frames.shape: {frames.shape}")

    #substractor1 = cv2.bgsegm.createBackgroundSubtractorMOG()    
    #substractor2 = cv2.createBackgroundSubtractorMOG2()
    #substractor3 = cv2.bgsegm.createBackgroundSubtractorGMG()
    #substractor4 = cv2.createBackgroundSubtractorKNN() 
    #substractor5 = cv2.bgsegm.createBackgroundSubtractorLSBP()
    #substractor6 = cv2.bgsegm.createBackgroundSubtractorCNT()

    
    i=0
    for frame in frames:

        if i==1200:
            
            sharp_img1 = substractor1.apply(frame)
            plt.imsave(f'prueba1_MOG.jpg', sharp_img1)

            '''
            sharp_img2 = substractor2.apply(frame)
            plt.imsave(f'prueba2.jpg', sharp_img2)

            sharp_img3 = substractor3.apply(frame)
            plt.imsave(f'prueba3.jpg', sharp_img3)

            sharp_img4 = substractor4.apply(frame)
            plt.imsave(f'prueba4.jpg', sharp_img4)

            sharp_img5 = substractor5.apply(frame)
            plt.imsave(f'prueba5.jpg', sharp_img5)

            sharp_img6 = substractor6.apply(frame)
            plt.imsave(f'prueba6.jpg', sharp_img6)
            

            sharp_img7 = substractor7.apply(frame)
            plt.imsave(f'prueba7.jpg', sharp_img7)
            '''

            break
            
        else: 
            substractor1.apply(frame)
            #substractor2.apply(frame)
            #substractor3.apply(frame)
            #substractor4.apply(frame)
            #substractor5.apply(frame)
            #substractor6.apply(frame)
            #substractor7.apply(frame)

        i+=1

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
     \n 6: rembg (special)
     """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--index', type=int, help=description, default=0)


    args = parser.parse_args()

    s_index = args.index




    vid_path = '/ghome/group07/test/AICity_data/train/S03/c010/vdo.avi'
    annotations_path = '/ghome/group07/test/ai_challenge_s03_c010-full_annotation.xml'
    #try_state_of_the_art(vid_path)
    state_of_the_art(vid_path, s_index)