""" Functions to create models """

import torch
import torch.nn as nn
import sys


def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes)
    elif model_name == '3d_resnet':
        return create_3d_res(load_pretrain, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

def create_3d_res(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model.blocks[-1].proj = nn.Identity()
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )

def create_x3d_xs(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )
    