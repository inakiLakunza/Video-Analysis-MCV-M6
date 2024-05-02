""" Functions to create models """

import torch
import torch.nn as nn

def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

def create_x3d_xs(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=True)

    #for param in list(model.parameters()):
    #    param.requires_grad = False

    model.blocks[5].proj = nn.Identity()        
    params_to_train = []
    
    model = nn.Sequential(model, nn.Linear(2048, num_classes, bias=True))
    

    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_train.append(param)

    return model,  params_to_train
    