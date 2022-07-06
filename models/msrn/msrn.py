import torch
from torch import nn
import math

# init ICNR to start from NN interpolation
def ICNR(tensor, scale_factor=2, initializer=nn.init.kaiming_normal_):
    print('Tensor shape: ' + str(tensor.shape))
    OUT, IN, H, W = tensor.shape
    sub = torch.zeros(math.ceil(OUT/scale_factor**2), IN, H, W)
    sub = initializer(sub)
    print('Sub shape: ' + str(sub.shape))
    kernel = torch.zeros_like(tensor)
    for i in range(OUT):
        kernel[i] = sub[i//scale_factor**2]
        
    return kernel

# residual module
class MSRB(nn.Module):
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()
        self.n_feats = n_feats
        self.conv3_1 = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=5, stride=1, padding=2)
        self.conv3_2 = nn.Conv2d(2*self.n_feats, 2*self.n_feats, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(2*self.n_feats, 2*self.n_feats, kernel_size=5, stride=1, padding=2)
        self.conv1_3 = nn.Conv2d(4*self.n_feats, self.n_feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x_input = x.clone()
        
        x3_1 = self.relu(self.conv3_1(x))
        x5_1 = self.relu(self.conv5_1(x))
        x1 = torch.cat([x3_1, x5_1], 1)
        
        x3_2 = self.relu(self.conv3_2(x1))
        x5_2 = self.relu(self.conv5_2(x1))
        x2 = torch.cat([x3_2, x5_2], 1)
        
        x_final = self.conv1_3(x2)
        return x_final + x_input
    
# Full structure
class MSRN_Upscale(nn.Module):
    def __init__(self, n_input_channels=3, n_blocks=8, n_feats=64, n_scale=4):
        super(MSRN_Upscale, self).__init__()
        
        self.n_blocks = n_blocks
        self.n_feats = n_feats
        self.n_scale = n_scale
        self.n_input_channels = n_input_channels
        
        # input
        self.conv_input = nn.Conv2d(self.n_input_channels, self.n_feats, kernel_size=3, stride=1, padding=1)
        
        # body
        conv_blocks = []
        for i in range(self.n_blocks):
            conv_blocks.append(MSRB(self.n_feats))
        self.conv_blocks = nn.Sequential(*conv_blocks)       
        
        self.bottle_neck = nn.Conv2d((self.n_blocks+1)* self.n_feats, self.n_feats, kernel_size=1, stride=1, padding=0)
              
        # tail
        self.conv_up = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1, bias=1)
        self.pixel_shuffle = nn.Upsample(scale_factor=self.n_scale, mode='nearest')
        self.conv_output = nn.Conv2d(self.n_feats, n_input_channels, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)

    
    def _init_pixel_shuffle(self):
        kernel = ICNR(self.conv_up.weight, scale_factor=self.n_scale)
        self.conv_up.weight.data.copy_(kernel)       
        
    def forward(self, x):
        x_input = x.clone()
        
        features=[]

        # M0
        x = self.conv_input(x)
        features.append(x)
        
        # body
        for i in range(self.n_blocks):
            x = self.conv_blocks[i](x)
            features.append(x)
            
        x = torch.cat(features, 1)
        
        x = self.bottle_neck(x)

        x = self.conv_up(x)
        x = self.pixel_shuffle(x)
        x = self.conv_output(x)
        
        return x