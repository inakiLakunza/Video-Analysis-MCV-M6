import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm

video_path = '/ghome/group07/test/AICity_data/train/S03/c010/vdo.avi'
annotations_path = './ai_challenge_s03_c010-full_annotation.xml'


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
        estimation = np.logical_or.reduce(estimation, axis=3)

    if color_space == 'HSV':
        #Only works with H and S
        new_frames = frames[:,:,:,:2]
        new_mean_ = mean_[:,:,:2]
        new_std_ =std_[:,:,:2]
        estimation = np.abs(new_frames - new_mean_) >= (alpha_ * (new_std_ + 2))
        estimation = np.logical_or.reduce(estimation, axis=3)

    if color_space == 'Lab':
        #Only works with H and S
        new_frames = frames[:,:,:,1:]
        new_mean_ = mean_[:,:,1:]
        new_std_ =std_[:,:,1:]
        estimation = np.abs(new_frames - new_mean_) >= (alpha_ * (new_std_ + 2))
        estimation = np.logical_or.reduce(estimation, axis=3)

    if color_space == 'YUV':
        #Only works with H and S
        new_frames = frames[:,:,:,1:]
        new_mean_ = mean_[:,:,1:]
        new_std_ =std_[:,:,1:]
        estimation = np.abs(new_frames - new_mean_) >= (alpha_ * (new_std_ + 2))
        estimation = np.logical_or.reduce(estimation, axis=3)
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

   

    # Closing
    #kernel = np.ones((35,35),np.uint8)
    #gray_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_CLOSE, kernel)
    #plt.imsave(f'./pruebas_3/before_{frame_idx + inc}_R.jpg', gray_frame[:,:,0], cmap='gray')
    #plt.imsave(f'./pruebas_3/before_{frame_idx + inc}_G.jpg', gray_frame[:,:,1], cmap='gray')
    #plt.imsave(f'./pruebas_3/before_{frame_idx + inc}_B.jpg', gray_frame[:,:,2], cmap='gray')
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
    
    # Compute metrics
    precision = utils.compute_ap(gt_mask_list, pred_mask_list)
    recall = utils.compute_ap(pred_mask_list, gt_mask_list)

    # plt.imsave(f'./pruebas_1_1/after_pred_mask_{frame_idx + inc}.jpg', pred_mask, cmap="gray")
    # plt.imsave(f'./pruebas_1_1/after_gt_mask_{frame_idx + inc}.jpg', gt_mask, cmap="gray")

    #plt.imsave(f'./pruebas_3/after_{frame_idx + inc}.jpg', rgb_frame)

    return precision, recall

    

def Gaussian_Estimation(video_path: str, annotations_path: str, alpha, color_space):
    gray_frames, color_frames, rgb_frames = utils.read_video(video_path, color_space)
    print(f"gray_frames.shape: {gray_frames.shape} \t color_frames.shape: {color_frames.shape}")

    gt = utils.read_annotations(annotations_path)

    # Split 25-75 frames
    gray_frames_25, gray_frames_75 = utils.split_frames(gray_frames)
    color_frames_25, color_frames_75 = utils.split_frames(color_frames)
    rgb_frames_25, rgb_frames_75 = utils.split_frames(rgb_frames)

    # Background modeling
    mean_, std_ = mean_and_std(color_frames_25)
    print(f"mean video: {mean_.shape} \t std video: {std_.shape}")

    estimation = estimate_foreground(color_frames_75, mean_, std_, alpha, color_space)
    print(f"estimation video: {estimation.shape}")

    # Separate objects and compute metrics
    precision_list = []
    recall_list = []
    for frame_idx in (pbar := tqdm(range(estimation.shape[0]))):
    #for frame_idx in (pbar := tqdm(range(1))):
        precision, recall = connected_components(frame_idx, color_frames_25.shape[0], estimation[frame_idx], color_frames_75[frame_idx], gt, rgb_frames_75[frame_idx])
        precision_list.append(precision)
        recall_list.append(recall)
        pbar.set_description(f"[alpha = {alpha}] Current AP {round(sum(precision_list) / len(precision_list), 2)}")

    AP = sum(precision_list) / len(precision_list)
    AR = sum(recall_list) / len(recall_list)

    print(f"Color space: {color_space}")
    print(f"[alpha = {alpha}] Average Preicison = {AP}")
    print(f"[alpha = {alpha}] Average Recall = {AR}")

    #utils.plot_list(color_frames_25.shape[0], AP)

    #utils.make_video(color_frames_75)
    #print(f"Video done.")



if __name__ == "__main__":
    alphas = [2,3,4,5,6,7]
    color_space = ['RGB','HSV','Lab','YUV']
    #for color in color_space:
    #for alpha in alphas:
    Gaussian_Estimation(video_path, annotations_path, 1, 'Lab')
    
    
