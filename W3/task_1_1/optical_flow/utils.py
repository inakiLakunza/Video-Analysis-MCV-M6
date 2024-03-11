import cv2


import numpy as np

import matplotlib.pyplot as plt
import os


def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()


#extracted from team 2 2023
def load_optical_flow(file_path: str):
    # channels arranged as BGR
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.double)
    return convert_optical_flow_to_image(img)


#Extracted from optical flowlib 
def convert_optical_flow_to_image(flow: np.ndarray) -> np.ndarray:


    img_u = (flow[:, :, 2] - 2 ** 15) / 64
    img_v = (flow[:, :, 1] - 2 ** 15) / 64

    img_available = flow[:, :, 0]  # whether a valid GT optical flow value is available
    img_available[img_available > 1] = 1

    img_u[img_available == 0] = 0
    img_v[img_available == 0] = 0

    optical_flow = np.dstack((img_u, img_v, img_available))
    return optical_flow


def OF_MSEN(GT, pred, output_dir: str, visualize=True):
    """
    Computes "Mean Square Error in Non-occluded areas"
    """

    u_diff, v_diff = GT[:, :, 0] - pred[:, :, 0], GT[:, :, 1] - pred[:, :, 1]
    se = np.sqrt(u_diff ** 2 + v_diff ** 2)
    sen = se[GT[:, :, 2] == 1]
    msen = np.mean(sen)

    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        
        se[GT[:, :, 2] == 0] = 0  # Exclude non-valid pixels
        plt.figure(figsize=(11, 4))
        img_plot = plt.imshow(se)
        img_plot.set_cmap("Blues")
        plt.title(f"Mean Square Error in Non-Occluded Areas")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, "OF_MSEN.png"))
        plt.clf()

    return msen, sen



def calculate_pepn(gt_flow, pred_flow, th=3):
    mask = gt_flow[:, :, 2] == 1 # mask of the valid points
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]

    return np.sum(sqrt_error_masked > th) / len(sqrt_error_masked)


def norm(x): 
    return (1 + ((x - x.mean()) / x.std())) / 2


def standarize(x): 
    return (x - x.min()) / (x.max() - x.min())


