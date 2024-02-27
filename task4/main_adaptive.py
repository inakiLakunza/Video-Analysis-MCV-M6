import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import utils_adaptive as utils
import tqdm 
from utils import *

from typing import *

import os



def mean_and_std(frames_25):
    """
    Compute the mean and std frame of the first 25% frames

    Parameters
        frames_25 : np.ndarray([536, 1080, 1920, 1])

    Returns
        mean : np.ndarray([1080, 1920])
        std : np.ndarray([1080, 1920])
    """
    return np.mean(frames_25, axis=0), np.std(frames_25, axis=0)


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


def estimate_foreground(frames, mean_, std_, alpha_, color_space):
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

    if color_space == 'RGB':
        #Works with 3 channels
        estimation = np.abs(frames - mean_) >= (alpha_ * (std_ + 2))
        estimation = np.logical_or.reduce(estimation, axis=2)

    if color_space == 'HSV':
        #Only works with H and S
        new_frames = frames[:,:,:2]
        new_mean_ = mean_[:,:,:2]
        new_std_ =std_[:,:,:2]
        estimation = np.abs(new_frames - new_mean_) >= (alpha_ * (new_std_ + 2))
        estimation = np.logical_or.reduce(estimation, axis=2)

    if color_space == 'Lab':
        #Only works with H and S
        new_frames = frames[:,:,1:]
        new_mean_ = mean_[:,:,1:]
        new_std_ =std_[:,:,1:]
        estimation = np.abs(new_frames - new_mean_) >= (alpha_ * (new_std_ + 2))
        estimation = np.logical_or.reduce(estimation, axis=2)

    if color_space == 'YUV':
        #Only works with H and S
        new_frames = frames[:,:,1:]
        new_mean_ = mean_[:,:,1:]
        new_std_ =std_[:,:,1:]
        estimation = np.abs(new_frames - new_mean_) >= (alpha_ * (new_std_ + 2))
        estimation = np.logical_or.reduce(estimation, axis=2)
    return estimation.astype(bool)


def connected_components(frame_idx, inc, gray_frame, color_frame, gt, rgb_frame):
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
    gray_frame = cv2.medianBlur(gray_frame, 3)
    # Closing
    kernel = np.ones((3,3),np.uint8)
    gray_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_CLOSE, kernel)
    gray_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_OPEN, kernel)

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


        if (area > 1_000) and (area < 250_000): 
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
            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
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
            cv2.rectangle(rgb_frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 3)
            cv2.rectangle(gt_mask, (xtl, ytl), (xbr, ybr), 255, -1)

            #plt.imsave(f'./pruebas_1_1/gt_mask_{len(gt_mask_list)}.jpg', gt_mask, cmap="gray")

            gt_mask_list.append(gt_mask)
    
    plt.imshow(rgb_frame)
    

    # Compute metrics
    AP = utils.compute_ap(gt_mask_list, pred_mask_list)
    

    return AP, rgb_frame



    

def Gaussian_Estimation(video_path: str, annotations_path: str):
    gray_frames, color_frames = utils.read_video(video_path)
    print(f"gray_frames.shape: {gray_frames.shape} \t color_frames.shape: {color_frames.shape}")

    gt = utils.read_annotations(annotations_path)

    # Split 25-75 frames
    gray_frames_25, gray_frames_75 = utils.split_frames(gray_frames)
    color_frames_25, color_frames_75 = utils.split_frames(color_frames)

    # Background modeling
    mean_, std_ = mean_and_std(gray_frames_25)
    print(f"mean video: {mean_.shape} \t std video: {std_.shape}")

    estimation = estimate_foreground(gray_frames_75, mean_, std_)
    print(f"estimation video: {estimation.shape}")

    # Separate objects
    for frame_idx in range(5):
        connected_components(frame_idx, color_frames_25.shape[0], estimation[frame_idx], color_frames_75[frame_idx], gt)

    
    return mean_, std_
    #utils.make_video(color_frames_75)
    #print(f"Video done.")
  

def fit_(video, N_iterations:int,  size:tuple, color_space):


    mean =  np.zeros(size)
    std = np.zeros(size)
    for idx in tqdm.tqdm(range(N_iterations), desc="Fitting the mean and standard deviation at the beggining"):

        _, frame = video.read()
        
        if color_space == 'RGB':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if color_space == 'HSV':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if color_space == 'Lab':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        if color_space == 'YUV':
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        deviation_avg = frame_rgb - mean

        mean += (deviation_avg / (idx + 1))
        deviation_frame = (frame_rgb - mean)

        std += (deviation_avg * deviation_frame)

    std = np.sqrt(std / (idx +1))

    print("Storing COmputed Mean and Standard deviation in the first 25% of the video")

    #cv2.imwrite(os.path.join('./images/pruebas', "mean_25.jpg"), mean.astype("uint8"))
    #cv2.imwrite(os.path.join('./images/pruebas', "std_25.jpg"), std.astype("uint8"))
    
    return mean , std



def Adaptive_Gaussian_Estimation(vid, annotations_path:str, mean:np.ndarray, std:np.ndarray, rho:float=0.25, alpha:int=2, color_space='RGB'):
    
    evol_means = []
    evol_std = []
    evol_foregrounds = []
    evol_ap = []


    # evolution of the means and the standard deviations
    evol_means.append(mean)
    evol_std.append(std)
    
    print(f"mean video: {mean.shape} \t std video: {std.shape}")
    
    init_frame_id = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
    
    # frames = 2141
    for idx in (pbar:= tqdm.tqdm(range(N_75), desc="Estimating Foreground")):
        _, color_frame = vid.read()
        frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        if color_space == 'RGB':
            frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        if color_space == 'HSV':
            frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        if color_space == 'Lab':
            frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2Lab)
        if color_space == 'YUV':
            frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2YUV)

        weights = (utils.calcular_pesos_exponenciales(len(evol_means))) if len(evol_means) <10 else utils.calcular_pesos_exponenciales(10)

        mean_with_to_estimate = utils.compute_weighted_avg(evol_means, weights) #np.mean(np.array(evol_means) * weights.reshape((len(weights), 1, 1)))
        std_with_to_estimate = utils.compute_weighted_avg(evol_std, weights) #evol_std[-1]#np.mean(np.array(evol_std) * weights.reshape((len(weights), 1, 1)))
        print(frame_rgb.shape)
        estimated_foregrounds = estimate_foreground(frames=frame_rgb, mean_=mean_with_to_estimate, std_=std_with_to_estimate, alpha_=alpha, color_space = color_space)
        #save_img(img=estimated_foregrounds*255, rho=rho, idx=idx, directorio = f'images/pruebas_foreground_{str(rho)}_{str(alpha)}')

       
        # Compute the mean  and std by the variations of the window
        estimated_foregrounds_3 = np.expand_dims(estimated_foregrounds, axis=-1)
        # Repite el array a lo largo de la dimensión del canal
        estimated_foregrounds_3 = np.repeat(estimated_foregrounds_3, 3, axis=-1)

        mean = np.where(estimated_foregrounds_3, mean_with_to_estimate, (rho * frame_rgb) + (1-rho) * mean_with_to_estimate)   
        #save_img(img=mean, rho=rho, idx=idx, directorio=f"images/pruebas_means_{str(rho)}_{str(alpha)}")

        std = np.where(estimated_foregrounds_3, std_with_to_estimate, np.sqrt(rho * (frame_rgb - mean) ** 2 + (1 - rho) * std_with_to_estimate ** 2))
        #save_img(img=std, rho=rho, idx=idx, directorio=f"images/pruebas_std_{str(rho)}_{str(alpha)}")
        
        # updating the evolution of the mean and the std
        evol_means.append(mean)
        evol_std.append(std)

        if len(evol_means) > 10:
            evol_means = evol_means[-10:]
            evol_std = evol_std[-10:]


        ## Computing metrrics
        ap, rgb_frame = connected_components(idx, inc=N_25, gray_frame=estimated_foregrounds, color_frame=color_frame, gt=gt, rgb_frame=rgb_frame)
        #save_img(img=rgb_frame, rho=rho, idx=idx, directorio=f"images/results_{str(rho)}_{str(alpha)}")


        evol_ap.append(ap)        
        pbar.set_description(f"[alpha = {alpha}] Current AP {round(sum(evol_ap) / len(evol_ap), 2)}")

        # Guardar la lista en el archivo usando pickle
    #with open(f'images/results/results_{str(rho)}_{str(alpha)}_ap.pkl', 'wb') as archivo :
        #pickle.dump(evol_ap, archivo)
    
    return evol_ap






if __name__ == "__main__":
    import pickle  
    import copy 

    video_path = '/ghome/group07/test/AICity_data/train/S03/c010/vdo.avi'
    annotations_path = './ai_challenge_s03_c010-full_annotation.xml'

        
    #for idx in [i / 10 for i in range(11)]:
    rho = 0.3
    print("Starting the fitting for rho: ", rho)

    vid = cv2.VideoCapture(video_path)
    N = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    SIZE = (H, W, 3)
    N_25 = int(0.25*N)
    N_75 = N-N_25

    
    print(f"gray_frames.shape: {N}")
    gt = utils.read_annotations(annotations_path)

    color = 'HSV'
    alpha = 6
    # Background modeling
    mean_, std_ = fit_(video=vid, N_iterations=N_25, size=SIZE,color_space = color)

    evol_ap = Adaptive_Gaussian_Estimation(vid, annotations_path, mean_, std_,  rho=rho, alpha=alpha, color_space=color )
    AP = sum(evol_ap) / len(evol_ap)
    print(f"Color space: {color}")
    print(f"[alpha = {alpha}] Average Preicison = {AP}")
