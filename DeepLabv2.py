import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, k_size, stride, padding, dilation=1, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias = False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu : 
        block.append(nn.ReLU(inplace=True))
    return nn.Sequential(*block)


class Bottleneck(nn.Module):
    def __init__(self,in_ch, out_ch, stride=1, dilation =1, downsample = False):
        super().__init__()
        self.block = nn.Sequential(conv_block(in_ch,out_ch//4,1,1,0),
                                  conv_block(out_ch//4, out_ch//4, 3, stride ,dilation, dilation),
                                  conv_block(out_ch//4,out_ch,1,1,0,False))
        self.downsample = nn.Sequential(
            conv_block(in_ch, out_ch, 1, stride, 0, 1, relu = False)) if downsample else None
        
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        out = self.block(x)
        identity = x
        if self.downsample is not None :
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)
        return out


class Resblock(nn.Module):
    def __init__(self,in_ch,out_ch, stride, dilation, num_layers):
        super().__init__()
        block = []
        for i in range(num_layers):
            block.append(Bottleneck(in_ch if i is 0 else out_ch,
                                   out_ch,
                                    stride if i is 0 else 1,
                                    dilation, 
                                    True if i is 0 else False
                                   ))
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        return self.block(x)
            
class ResNet101(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()
        self.block = nn.Sequential(conv_block(in_channels, 64,7,2,3,1,True),
                                  nn.MaxPool2d(3,2,padding=1),
                                  Resblock(64,256,1,1,3),
                                  Resblock(256,512,2,1,4),
                                  Resblock(512,1024,1,2,23),
                                  Resblock(1024,2048,1,4,3)
                                  )
    def forward(self,x):
        return self.block(x)
    

class ASPP(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block1 = nn.Sequential(conv_block(in_ch,in_ch,3,1,padding=6,dilation=6),
                                   conv_block(in_ch,in_ch,1,1,0),
                                   conv_block(in_ch,out_ch,1,1,0))
        self.block2 = nn.Sequential(conv_block(in_ch,in_ch,3,1,padding=12,dilation=12),
                                   conv_block(in_ch,in_ch,1,1,0),
                                   conv_block(in_ch,out_ch,1,1,0))
        self.block3 = nn.Sequential(conv_block(in_ch,in_ch,3,1,padding=18,dilation=18),
                                   conv_block(in_ch,in_ch,1,1,0),
                                   conv_block(in_ch,out_ch,1,1,0))
        self.block4 = nn.Sequential(conv_block(in_ch,in_ch,3,1,padding=24,dilation=24),
                                   conv_block(in_ch,in_ch,1,1,0),
                                   conv_block(in_ch,out_ch,1,1,0))
    
    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        return out1+out2+out3+out4

class DeepLabV2(nn.Module):
    def __init__(self,in_channels, num_classes):
        super().__init__()
        self.backbone = ResNet101(in_channels)
        self.aspp = ASPP(2048,num_classes)
    
    def forward(self,x):
        out = self.backbone(x)
        out = self.aspp(out)
        
        out = F.interpolate(out , size = (x.shape[-1],x.shape[-2]), mode = 'bilinear', align_corners=True)
        return out
