import cv2
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from PyFlow import demo as pyflow_query
from RAFT import main as raft_query
from OpticalFlowToolkit.lib import flowlib # https://github.com/liruoteng/OpticalFlowToolkit
import flow_vis

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


# MISC ============================================================

def visualize_flowvis(im1_path: str, flow, filename):
    flow_color = flow_vis.flow_to_color(flow[:, :, :2], convert_to_bgr=True)
    cv2.imwrite(f'./results/flow_vis_{filename}.png', flow_color)

def visualize_magdir(im1_path: str, flow, filename):
    im1 = np.array(Image.open(im1_path).convert('RGB'))
    im1 = im1.astype(float) / 255.
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f'./results/magnitude_dir_{filename}.png', bgr)


def visualize_arrow(im1_path: str, flow, filename):
    im1 = np.array(Image.open(im1_path).convert('RGB'))
    im1 = im1.astype(float) / 255.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Background image
    ax.imshow(im1)
    step = 10 # tune this
    x, y = np.meshgrid(np.arange(0, flow.shape[1], step), np.arange(0, flow.shape[0], step))
    u = flow[y, x, 0]
    v = flow[y, x, 1]
    
    quiver = ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, width=0.0015, headwidth=5)    

    plt.savefig(f'./results/arrow_{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_direction_idx_plot(im1_path: str, flow, filename):
    im1 = np.array(Image.open(im1_path).convert('RGB'))
    im1 = im1.astype(float) / 255.
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Background
    ax.imshow(im1)

    step = 10
    x, y = np.meshgrid(np.arange(0, flow.shape[1], step), np.arange(0, flow.shape[0], step))
    u = flow[y, x, 0]
    v = flow[y, x, 1]
    
    direction = np.arctan2(v, u)
    norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
    cmap = plt.cm.hsv
    colors = cmap(norm(direction))
    colors = colors.reshape(-1, colors.shape[-1])
    quiver = ax.quiver(x, y, u, v, color=colors, angles='xy', scale_units='xy', scale=1, width=0.0015, headwidth=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for colorbar
    cbar = fig.colorbar(sm, ax=ax, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    cbar.set_label('Direction')
    plt.savefig(f'./results/{filename}_direction.png', dpi=300, bbox_inches='tight')
    plt.close()




if __name__ == '__main__':
    im1_path = '../data_stereo_flow/training/colored_0/000045_10.png'
    im2_path = '../data_stereo_flow/training/colored_0/000045_11.png'
    gt_path = '../data_stereo_flow/training/flow_noc/000045_10.png'

    flow_gt = flowlib.read_flow(gt_path)
    
    # Optical flows
    flow_pyflow, ela_pyflow = compute_pyflow(im1_path, im2_path)
    flow_raft, ela_raft = compute_raft(im1_path, im2_path)
    print(f"\nTIME (PyFlow): {ela_pyflow:.4f}s")
    print(f"TIME (RAFT): {ela_raft:.4f}s")

    # MSEN
    msen_pyflow = calculate_msen(flow_gt, flow_pyflow)
    msen_raft = calculate_msen(flow_gt, flow_raft)
    print(f"\nMSEN (PyFlow): {msen_pyflow:.4f}")
    print(f"MSEN (RAFT): {msen_raft:.4f}")

    # PEPN
    pepn_pyflow = calculate_pepn(flow_gt, flow_pyflow)
    pepn_raft = calculate_pepn(flow_gt, flow_raft)
    print(f"\nPEPN (PyFlow): {pepn_pyflow * 100:.2f}%")
    print(f"PEPN (RAFT): {pepn_raft * 100:.2f}%")

    # Save results
    visualize_flowvis(im1_path, flow_gt, filename="GT")
    visualize_flowvis(im1_path, flow_pyflow, filename="PyFlow")
    visualize_flowvis(im1_path, flow_raft, filename="RAFT")

