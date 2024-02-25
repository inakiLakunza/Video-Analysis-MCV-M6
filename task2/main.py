import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import utils
import tqdm 


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
    estimation = np.abs(frames - mean_) >= (alpha_ * (std_ + 2))
    return estimation.astype(bool)


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

    plt.imsave(f'./pruebas_1_1/before_{frame_idx + inc}.jpg', gray_frame, cmap='gray')

    # Opening
    kernel = np.ones((3,3),np.uint8)
    gray_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_OPEN, kernel)

    # Connected components
    analysis = cv2.connectedComponentsWithStats(gray_frame, 4, cv2.CV_32S) 
    (totalLabels, label_ids, values, centroid) = analysis 
    output = np.zeros(gray_frame.shape, dtype="uint8")

    # Create mask1 and mask2 to compute metrics
    pred_mask = np.zeros(gray_frame.shape, dtype="uint8") # prediction
    gt_mask = np.zeros(gray_frame.shape, dtype="uint8") # gt

    # Loop through each component 
    for i in range(1, totalLabels): 
        area = values[i, cv2.CC_STAT_AREA] 

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

    # Paint the GT Bounding boxes
    real_frame_idx = str(frame_idx + inc)
    if real_frame_idx in gt:
        for box in gt[real_frame_idx]:
            xtl = int(float(box['xtl']))
            ytl = int(float(box['ytl']))
            xbr = int(float(box['xbr']))
            ybr = int(float(box['ybr']))
            
            cv2.rectangle(color_frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 3)
            cv2.rectangle(gt_mask, (xtl, ytl), (xbr, ybr), 255, -1)
    
    IoU = binaryMaskIOU(pred_mask, gt_mask)
    print(f"Frame {real_frame_idx} IoU = {IoU}")

    plt.imsave(f'./pruebas_1_1/after_pred_mask_{frame_idx + inc}.jpg', pred_mask, cmap="gray")
    plt.imsave(f'./pruebas_1_1/after_gt_mask_{frame_idx + inc}.jpg', gt_mask, cmap="gray")

    plt.imsave(f'./pruebas_1_1/after_{frame_idx + inc}.jpg', color_frame)


def binaryMaskIOU(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1 == 255, mask2 == 255))
    union = mask1_area+mask2_area-intersection
    if union == 0: # Evitar dividir entre 0
        return 0
    iou = intersection/(union)
    return iou


def MeanIOU(gt_dict, pred_dict):
    result = 0
    for name, mask in gt_dict.items():
        result += binaryMaskIOU(gt_dict[name], pred_dict[name])
    
    result /= len(gt_dict)
    return result
    

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
    
    
def Adaptive_Gaussian_Estimation(video_path: str, annotations_path:str, _window:int=4, rho:float=0.25, extract_by_mean:bool = False):
    
    evol_means = []
    evol_std = []
    print("gola")
    gray_frames, color_frames = utils.read_video(video_path, max_frames=500)
    
    
    print(f"gray_frames.shape: {gray_frames.shape} \t color_frames.shape: {color_frames.shape}")
    gt = utils.read_annotations(annotations_path)
    
    # Split 25-75 frames
    gray_frames_25, gray_frames_75 = utils.split_frames(gray_frames)
    color_frames_25, color_frames_75 = utils.split_frames(color_frames)


    # Background modeling
    mean_, std_ = mean_and_std(gray_frames_25)

    # evolution of the means and the standard deviations
    evol_means.append(mean_)
    evol_std.append(std_)
    
    print(f"mean video: {mean_.shape} \t std video: {std_.shape}")
    
    # frames = 2141
    for idx in tqdm.tqdm(range(0, gray_frames_75.shape[0], _window), desc="Estimating Foreground"):
        if idx + 10 > gray_frames_75.shape[0]:
            set_indexes = np.arange(start=idx, stop=gray_frames_75.shape[0])
        else: 
            set_indexes = np.arange(start=idx, stop=idx+_window)
    
        # get the last adapated mean and std
        last_mean = evol_means[-1]
        last_std = evol_std[-1] 
        
        if extract_by_mean is True:
            I = np.mean(gray_frames_75[set_indexes])
        else:
            I = (gray_frames_75[set_indexes[-1]])
        
        
        # SUM all the variations over the window
        estimated_foregrounds = estimate_foreground(frames=I, mean_=last_mean, std_=last_std)
        
        # Compute the mean  and std by the variations of the window
        mean = np.where(estimated_foregrounds, last_mean, (rho * I) + (1-rho) * last_mean)   
        std = np.where(estimated_foregrounds, last_std, np.sqrt(rho * (I - mean) ** 2 + (1 - rho) * last_std ** 2))
        
        # updating the evolution of the mean and the std
        evol_means.append(mean)
        evol_std.append(std)
    
    return evol_means, evol_std



def compute_masks(frames:np.ndarray, mean: Optional[List[np.ndarray]], std: Optional[List[np.ndarray]]):
    assert type(mean) == type(std), "Missmatch in types"
    
    if isinstance(mean, list):
        pass
    else:
        pass





if __name__ == "__main__":
    
    video_path = './AICity_data/train/S03/c010/vdo.avi'
    annotations_path = './ai_challenge_s03_c010-full_annotation.xml'

    evol_means, evol_std, frames, estimated_foregrounds = Adaptive_Gaussian_Estimation(video_path, annotations_path)
    
        
    # Verificar si el directorio "pruebas" existe, si no, crearlo
    if not os.path.exists('./images/pruebas'):
        os.makedirs('./images/pruebas')

    # Guardar cada imagen en el directorio "pruebas"
    for i, img in enumerate(evol_means):
        filename = f'prueba_mean{i}.png'  # Nombre del archivo de imagen
        #cv2.imwrite(os.path.join('./images/pruebas', filename), img.astype("uint8"))


    # Guardar cada imagen en el directorio "pruebas"
    for i, img in enumerate(evol_std):
        filename = f'prueba_std{i}.png'  # Nombre del archivo de imagen
        #cv2.imwrite(os.path.join('./images/pruebas', filename), img.astype("uint8"))
        
        
        # Guardar cada imagen en el directorio "pruebas"
    #for i, img in enumerate(estimated_foregrounds):
    filename = f'prueba_frames{i}.png'  # Nombre del archivo de imagen
    cv2.imwrite(os.path.join('./images/pruebas', filename), (estimated_foregrounds*255).astype("uint8"))
    print("Imágenes guardadas en el directorio 'pruebas'.")