
import torch
from torch import nn
from torchvision import models
from torch.autograd import Variable



def buildX3DRGBOF(new_in_channels, load_pretrain, num_classes):
    
    backbone = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    
    # MODIFY INPUT LAYER
    #==================================================
    old_in_layer = backbone.blocks[0].conv.conv_t
    old_in_layer_weights = old_in_layer.weight.clone()

    # Creating new Conv3d layer
    new_layer = nn.Conv3d(in_channels=new_in_channels, 
                    out_channels=old_in_layer.out_channels, 
                    kernel_size=old_in_layer.kernel_size, 
                    stride=old_in_layer.stride, 
                    padding=old_in_layer.padding,
                    bias=old_in_layer.bias,
                    groups=old_in_layer.groups).requires_grad_()


    new_layer.weight[:, :old_in_layer.in_channels, :, :].data[...] = Variable(old_in_layer_weights, requires_grad=True)


    backbone.blocks[0].conv.conv_t = new_layer
    #==================================================


    # TAKE OUT LAST LAYER AND ADEQUATE IT
    #==================================================
    backbone.blocks[5].proj = nn.Identity()
    # # freeze layers
    # for param in backbone.parameters():
    #     param.requires_grad = False

    # WE WANT TO HAVE A GRADIENT ON THE INPUT LAYER
    backbone.blocks[0].conv.conv_t.requires_grad_ = True

    return nn.Sequential(
        backbone,
        nn.Linear(2048, num_classes, bias=True),
    )
    #==================================================


def buildResNetRGBOF(new_in_channels, load_pretrain, num_classes):
    
    backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    
    # MODIFY INPUT LAYER
    #==================================================
    old_in_layer = backbone.blocks[0].conv
    old_in_layer_weights = old_in_layer.weight.clone()

    # Creating new Conv3d layer
    new_layer = nn.Conv3d(in_channels=new_in_channels, 
                    out_channels=old_in_layer.out_channels, 
                    kernel_size=old_in_layer.kernel_size, 
                    stride=old_in_layer.stride, 
                    padding=old_in_layer.padding,
                    bias=old_in_layer.bias,
                    groups=old_in_layer.groups)


    new_layer.weight[:, :old_in_layer.in_channels, :, :].data[...] = Variable(old_in_layer_weights, requires_grad=True)

    backbone.blocks[0].conv = new_layer
    #==================================================



    # TAKE OUT LAST LAYER AND ADEQUATE IT
    #==================================================
    backbone.blocks[-1].proj = nn.Identity()
    # # freeze layers
    # for param in backbone.parameters():
    #     param.requires_grad = False

    # WE WANT TO HAVE A GRADIENT ON THE INPUT LAYER
    backbone.blocks[0].conv.requires_grad_ = True

    return nn.Sequential(
        backbone,
        nn.Linear(2048, num_classes, bias=True),
    )

    #==================================================

def buildAddedLayerResNetRGBOF(new_in_channels, load_pretrain, num_classes):
    
    backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    
    # INPUT SHAPE: (batch_size, 5[RGB_3+OF_2], 8[len of clip-1, 224, 224])
    # UPSAMPLE CHANNELS AND DOWNSAMPLE AFTERWARDS TO 3
    inverted_bottleneck = nn.Sequential(
        nn.Conv3d(new_in_channels, 16, kernel_size=(1, 7, 7), stride=(1, 1, 1), bias=False),
        nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv3d(16, 3, kernel_size=1, stride=1),
        nn.BatchNorm3d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU()
    )



    # TAKE OUT LAST LAYER AND ADEQUATE IT
    #==================================================
    backbone.blocks[-1].proj = nn.Identity()
    # # freeze layers
    # for param in backbone.parameters():
    #     param.requires_grad = False

    # WE WANT TO HAVE A GRADIENT ON THE INPUT LAYER
    backbone.blocks[0].conv.requires_grad_ = True

    return nn.Sequential(
        inverted_bottleneck,
        backbone,
        nn.Linear(2048, num_classes, bias=True),
    )

    #==================================================



def create(model_name: str, load_pretrain: bool, num_classes: int, new_in_channels: int =5) -> nn.Module:
    if model_name == 'x3d_xs':
        return buildX3DRGBOF(new_in_channels, load_pretrain, num_classes)
    elif model_name == '3d_resnet':
        return buildResNetRGBOF(new_in_channels, load_pretrain, num_classes)
    elif model_name == 'modified_resnet':
        return buildAddedLayerResNetRGBOF(new_in_channels, load_pretrain, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported")

