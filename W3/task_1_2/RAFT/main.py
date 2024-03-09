from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import cv2

"""
Example taken from PyTorch docs: https://pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_raft_pipeline(im1_path: str, im2_path: str, channels: int = 3, **kwargs):
    # read images
    im1 = cv2.imread(im1_path, cv2.COLOR_BGR2RGB)
    im2 = cv2.imread(im2_path, cv2.COLOR_BGR2RGB)
    print(im1.shape)
    original_size = im1.shape[:2]
    if channels == 1:
        im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
        im1 = im1.reshape([im1.shape[0], im1.shape[1], 1])
        im2 = im2.reshape([im2.shape[0], im2.shape[1], 1])
    
    # prepocess
    transforms = T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),
        T.Resize(size=(384, 1248)) # Multiples of 8! (Defauled to (520, 960))
    ])
    im1 = transforms(im1).unsqueeze(0).to(device)
    im2 = transforms(im2).unsqueeze(0).to(device)

    # pred
    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()

    list_of_flows = model(im1, im2)
    predicted_flows = list_of_flows[-1].cpu()

    resized_flow = torch.nn.functional.interpolate(predicted_flows, size=original_size, mode='bilinear', align_corners=False)
    resized_flow = resized_flow.detach().cpu().squeeze()
    resized_flow_np = resized_flow.permute(1, 2, 0).numpy()

    return resized_flow_np

