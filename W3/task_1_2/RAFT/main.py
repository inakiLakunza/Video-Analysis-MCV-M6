from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image

"""
Example taken from PyTorch docs: https://pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html
"""

# Prepare images
curr_img_og = Image.open('../../data_stereo_flow/training/colored_0/000045_10.png').convert('RGB')
ref_img_og = Image.open('../../data_stereo_flow/training/colored_0/000045_11.png').convert('RGB')
gt = Image.open('../../data_stereo_flow/training/flow_noc/000045_10.png')

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),
            T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch

device = "cuda" if torch.cuda.is_available() else "cpu"

curr_img = preprocess(curr_img_og).unsqueeze(0).to(device)
ref_img = preprocess(ref_img_og).unsqueeze(0).to(device)

print(f"shape = {curr_img.shape}, dtype = {curr_img.dtype}")

model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

list_of_flows = model(curr_img, ref_img)
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")

predicted_flows = list_of_flows[-1]
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

flow_imgs = flow_to_image(predicted_flows.squeeze().cpu())

# Plot optical flow
transforms = T.Compose(
        [
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Resize(size=(520, 960)),
        ]
    )
curr_img_np = transforms(np.array(curr_img_og)).permute(1, 2, 0)
flow_imgs_np = [np.array(img) for img in flow_imgs]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(curr_img_np)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(flow_imgs_np[0])
axes[1].set_title('Optical Flow')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('result.png')
plt.close(fig)

# Save og optical flow
original_size = curr_img_og.size[::-1]
resized_flow = torch.nn.functional.interpolate(predicted_flows, size=original_size, mode='bilinear', align_corners=False)

resized_flow = resized_flow.detach()
resized_flow = resized_flow.squeeze().permute(1, 2, 0)
resized_flow_np = resized_flow.cpu().numpy()
np.save('../results/Raft.npy', resized_flow_np)
