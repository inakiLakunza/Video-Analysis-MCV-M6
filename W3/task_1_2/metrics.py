import numpy as np
from PIL import Image
from OpticalFlowToolkit.lib import flowlib
# ref: https://github.com/liruoteng/OpticalFlowToolkit

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

def calculate_msen(gt_flow, pred_flow):
    """
    Function to compute  the Mean Square Error in Non-occluded areas
    gt_flow: the ground thruth optical flow
    pred_flow: the predicted optical flow
    """
    # Get the mask of the valid points
    mask = gt_flow[:, :, 2] == 1

    # compute the error in du and dv
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]
    msen = np.mean(sqrt_error_masked)

    return msen

gt_path = '../data_stereo_flow/training/flow_noc/000045_10.png'
flow_gt = flowlib.read_flow(gt_path)

print(f"GT Optical Flow (noc): {flow_gt.shape}")
print(flow_gt)

pyflow_flow = np.load('./results/PyFlow.npy')
print(f"PyFlow Optical Flow: {pyflow_flow.shape}")
print(pyflow_flow)

raft_flow = np.load('./results/Raft.npy')
print(f"Raft Optical Flow: {raft_flow.shape}")
print(raft_flow)

# MSEN
msen_pyflow = calculate_msen(flow_gt, pyflow_flow)
msen_raft = calculate_msen(flow_gt, raft_flow)

print(f"MSEN (PyFlow): {msen_pyflow:.4f}")
print(f"MSEN (RAFT): {msen_raft:.4f}")

# PEPN

