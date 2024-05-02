import torch
import torch.nn as nn


class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, input):
        return input.permute(self.dims).contiguous()

class ConvBN(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False),
            nn.BatchNorm2d(out_channels),
        )
class HCNBlock(nn.Sequential):
    """Extracts hierarchical co-occurrence feature from an input skeleton
    sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data.
        out_channels (int): Number of channels produced by the convolution.
        num_joints (int): Number of joints in each skeleton.
        with_bn (bool): Whether to append a BN layer after conv1.

    Shape:
        - Input: Input skeleton sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Output: Output feature map in :math:`(N, out_channels, T_{out},
            C_{out})` format

        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of joints,
            :math:`C_{out}` is the output size of the coordinate dimension.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 num_joints=25,
                 with_bn=False):
        inter_channels = out_channels // 2
        conv1 = ConvBN if with_bn else nn.Conv2d
        super().__init__(
            # conv1
            conv1(in_channels, out_channels, 1),
            nn.ReLU(),
            # conv2
            nn.Conv2d(out_channels, inter_channels, (3, 1), padding=(1, 0)),
            Permute((0, 3, 2, 1)),

            # conv3
            nn.Conv2d(num_joints, inter_channels, 3, padding=1),
            nn.MaxPool2d(2, stride=2),

            # conv4
            nn.Conv2d(inter_channels, out_channels, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.5),
        )
        
        
        

        
class JointModel(nn.Module):
    
    def __init__(self, in_channels:int, num_joints:int, num_classes:int, with_bn:bool=False):
        
                
        
        """
           - Input: :math:`(N, in_channels, T, V, M)`
            - Output: :math:`(N, D)` where
            :math:`N` is a batch size,
            :math:`T` is a length of input sequence,
            :math:`V` is the number of joints,
            :math:`M` is the number of instances in a frame.
        """
        
        super(JointModel, self).__init__()
        
        
        self.num_classes = num_classes
        self.net_l = HCNBlock(in_channels, 64, num_joints, with_bn)
        self.net_m = HCNBlock(in_channels, 64, num_joints, with_bn)
        
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        )
        
        self._global_pooling = nn.AdaptiveMaxPool2d(1)
        self.drop6 = nn.Dropout(p=0.5)
        
        
        self._decission_head = nn.Linear(in_features=256*2, out_features=num_classes)
        
        
        self._max_magnitude_displacement = nn.Linear(num_joints, 128, bias=True) 
        self._mean_magnitude_displacement = nn.Linear(num_joints, 128, bias=True)

    def forward(self, skeleton, skeleton_motion, magnitude):
        
        #n, m, t, v, c (my input)

        n, m, t, v, c = skeleton.size()
        skeleton= skeleton.permute(0, 1, 4, 2, 3).contiguous()  # N M C T V
        skeleton = skeleton.view(-1, c, t, v)  # N*M C T V
        
        skeleton_motion = skeleton_motion.permute(0, 1, 4, 2, 3).contiguous()
        skeleton_motion = skeleton_motion.view(-1, c, t, v)

        out_l = self.net_l(skeleton)
        out_m = self.net_m(skeleton_motion)
        
        ## from magnitude extract mean distances and maximun displacement from the maximun stimulus
        maximun_stimulus = torch.max(magnitude, dim=1)[0]
        max_displace = torch.max(maximun_stimulus, dim=1)[0]
        mean_displace = torch.mean(maximun_stimulus, dim=1)

        out_magnitude_max = self._max_magnitude_displacement(max_displace)
        out_magnitude_mean = self._mean_magnitude_displacement(mean_displace)
        
        trained_magnitude = torch.cat((out_magnitude_max, out_magnitude_mean), dim=1)
        
        out = torch.cat((out_l, out_m), dim=1)
        

        out = self.conv5(out)
        out = self.conv6(out)
        out = self._global_pooling(out)

        out = out.view(n, m, -1)  # N M D

        final_features = out.max(dim=1)[0]  # N D
        final_features = self.drop6(final_features)
        
        # combined features with skeleton motion and magnitude
        final_features  = torch.cat((final_features, trained_magnitude), dim=1)

        ## decission head        
        out = self._decission_head(final_features)
        
        
        return  out, final_features
    
    
if __name__ == "__main__":
    joints = torch.rand(size=(8, 2, 32, 17, 3))
    motion = torch.rand(size=(8, 2, 32, 17, 3))
    magnitude = torch.rand(size=(8, 4, 17))
    model = JointModel(in_channels=3, num_joints=joints.shape[3], num_classes=51)
    
    model(joints, motion, magnitude)
    
    