""" Functions to create models """

import torch
import torch.nn as nn

def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes)
    elif model_name == 'inception':
        return create_inception(load_pretrain, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

def create_inception(load_pretrain, num_classes):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    # freeze
    for param in model.parameters():
        param.requires_grad = False
    # last layer modified
    model.fc = nn.Linear(2048, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True
    return model

def create_x3d_xs(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )
    