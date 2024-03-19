import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import random

class Dataset():
    def __init__(self, path: str):
        # path = 
        self.path = path
        self.images = []
        self.labels = []
        self.captions = []

        self.categories = os.listdir(self.path)
        ctr = 0
        for cat in sorted(self.categories):
            regexp = os.path.join(self.path, cat, '*.jpg')
            imgs = sorted(glob.glob(regexp))
            for img in imgs:
                self.images.append(img)
                self.labels.append(ctr)
                self.captions.append(cat)
            ctr += 1
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        label = self.labels[index]
        caption = self.captions[index]
        return img_tensor, torch.tensor(label), caption

        


class WrapperDataloader(torch.utils.data.Dataset):
    def __init__(self, inner_dataset):
        self.inner_dataset = inner_dataset

    def __getitem__(self, idx):
        inputs, targets, _ = self.inner_dataset[idx]
        return inputs, targets

    def __iter__(self):
        for item in self.inner_dataset:
            yield item

    def __len__(self):
        return len(self.inner_dataset)