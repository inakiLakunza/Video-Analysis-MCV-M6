import time
import numpy as np
from PIL import Image
from PyFlow import demo as pyflow_query
from RAFT import main as raft_query
from OpticalFlowToolkit.lib import flowlib # https://github.com/liruoteng/OpticalFlowToolkit



""" From the Kitty Development Benchmark Suite:

Data format:
============

Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
contains the u-component, the second channel the v-component and the third
channel denotes if a valid ground truth optical flow value exists for that
pixel (1 if true, 0 otherwise). To convert the u-/v-flow into floating point
values, convert the value to float, subtract 2^15 and divide the result by 64:

flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
valid(u,v)  = (bool)I(u,v,3);
"""

# MODELS ===========================================================

def compute_pyflow(im1_path: str, im2_path: str):
    channels = 1
    kwargs = {
        'alpha': 0.012,
        'ratio': 0.75,
        'minWidth': 20,
        'nOuterFPIterations': 7,
        'nInnerFPIterations': 1,
        'nSORIterations': 30,
    }
    s = time.time()
    flow = pyflow_query.compute_flow_pipeline(im1_path, im2_path, channels, **kwargs)
    e = time.time()
    return flow, (e - s)

def compute_raft(im1_path: str, im2_path: str):
    channels = 3
    kwargs = {}
    s = time.time()
    flow = raft_query.compute_raft_pipeline(im1_path, im2_path, channels, **kwargs)
    e = time.time()
    return flow, (e - s)


# METRICS ===========================================================

def calculate_msen(gt_flow, pred_flow):
    mask = gt_flow[:, :, 2] == 1 # mask of the valid points
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]
    msen = np.mean(sqrt_error_masked)

    return msen

def calculate_pepn(gt_flow, pred_flow, th=3):
    mask = gt_flow[:, :, 2] == 1 # mask of the valid points
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]

    return np.sum(sqrt_error_masked > th) / len(sqrt_error_masked)



if __name__ == '__main__':
    im1_path = '../data_stereo_flow/training/colored_0/000045_10.png'
    im2_path = '../data_stereo_flow/training/colored_0/000045_11.png'
    gt_path = '../data_stereo_flow/training/flow_noc/000045_10.png'

    flow_gt = flowlib.read_flow(gt_path)
    
    # Optical flows
    flow_pyflow, ela_pyflow = compute_pyflow(im1_path, im2_path)
    raft_flow, ela_raft = compute_raft(im1_path, im2_path)
    print(f"\nTIME (PyFlow): {ela_pyflow:.4f}s")
    print(f"TIME (RAFT): {ela_raft:.4f}s")

    # MSEN
    msen_pyflow = calculate_msen(flow_gt, flow_pyflow)
    msen_raft = calculate_msen(flow_gt, raft_flow)
    print(f"\nMSEN (PyFlow): {msen_pyflow:.4f}")
    print(f"MSEN (RAFT): {msen_raft:.4f}")

    # PEPN
    pepn_pyflow = calculate_pepn(flow_gt, flow_pyflow)
    pepn_raft = calculate_pepn(flow_gt, raft_flow)
    print(f"PEPN (PyFlow): {pepn_pyflow * 100:.2f}%")
    print(f"PEPN (RAFT): {pepn_raft * 100:.2f}%")

