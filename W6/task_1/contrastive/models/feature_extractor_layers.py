import torch
import torch.nn as nn



class ImageFeatureExtractor2d(nn.Module):
    pass

  
class ImageFeatureExtractor3D(nn.Module):
    
    def __init__(self, name_model: str="x3d_xs", freeze:bool = True, linear:bool=True):
        super(ImageFeatureExtractor3D, self).__init__()
        
        self._freeze = freeze
        if name_model == "x3d_xs":
            if linear:
                self._feature_extractor = self.create_x3d_xs_linear(load_pretrained=True)
            else:
                self._feature_extractor = self.create_x3d_xs(load_pretrained=True)

                
        
        elif name_model == "resnet":
            if linear:
                self._feature_extractor = self.create_3d_resnet_linear(load_pretrained=True)
            
            else:
                self._feature_extractor = self.create_3d_resnet(load_pretrained=True)
            
        
    def create_x3d_xs_linear(self, load_pretrained:bool =True):
        model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrained)
        model.blocks[5].proj = nn.Identity()
        
        if self._freeze:
            for params in list(model.parameters()):
                params.requires_grad = False
                
        new_model = nn.Sequential(
        model,
        nn.Linear(2048, 512, bias=True),
        )        
          
                
        return new_model
    
    def create_x3d_xs(self, load_pretrained:bool =True):
        model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrained)
        model.blocks[5].proj = nn.Identity()
        
        if self._freeze:
            for params in list(model.parameters()):
                params.requires_grad = False
                

                
        return model

    def create_3d_resnet(self, load_pretrained:bool=True):
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=load_pretrained)

        model.blocks[-1].proj = nn.Identity()
     
        return model
    
    
    def create_3d_resnet_linear(self, load_pretrained=True):
    
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=load_pretrained)

        model.blocks[-1].proj = nn.Identity()
        

        new_model = nn.Sequential(
        model,
        nn.Linear(2048, 512, bias=True),
    )        
                
        return new_model
    
    
    def forward(self, x):
        return self._feature_extractor(x)