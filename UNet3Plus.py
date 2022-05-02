import numpy as np
import torch
import torch.nn as nn

class UNet3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=11):
        super(UNet3Plus, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        """
        https://arxiv.org/pdf/2004.08790.pdf
        논문의 그림을 봐야 이해가 빠릅니다. (Fig. 1)
        encoder node 5개 (e1-e5), decoder node 4개 (d4-d1)로 이루어져 있습니다.
        encoder node의 생성은 UNet의 방식과 같습니다 - 이전의 encoder node가 다음 encoder node를 생성합니다.
        하지만 각각의 decoder node들은 총 5개의 node들을 concat해 생성됩니다.
        총 5개의 '레이어'가 있고, 각각의 레이어에서 encoder node를 가져오되, 
        해당 레이어에 이미 생성된 decoder node가 존재한다면 encoder node 대신에 decoder node를 사용하는 방식입니다. 
        생성할 decoder node와 같은 레이어에 있는 encoder node를 제외하면 모두 height, width가 다르기 때문에,
        nn.MaxPool2d()와 nn.Upsample()을 이용해 height, width를 맞춰줍니다.
        5개의 node들의 channel도 64로 변경해줍니다. 
        64*5=340인 channel을 64로 줄여주면 decoder node가 생성됩니다.
        """
        
        # Encoder
        
        # e1
        self.e1_conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.e1_bn1 = nn.BatchNorm2d(64)
        self.e1_relu1 = nn.ReLU(inplace=True)
        self.e1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.e1_bn2 = nn.BatchNorm2d(64)
        self.e1_relu2 = nn.ReLU(inplace=True)
        
        # e2
        self.e2_maxpool = nn.MaxPool2d(kernel_size=2)
        self.e2_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.e2_bn1 = nn.BatchNorm2d(128)
        self.e2_relu1 = nn.ReLU(inplace=True)
        self.e2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.e2_bn2 = nn.BatchNorm2d(128)
        self.e2_relu2 = nn.ReLU(inplace=True)
        
        # e3
        self.e3_maxpool = nn.MaxPool2d(kernel_size=2)
        self.e3_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.e3_bn1 = nn.BatchNorm2d(256)
        self.e3_relu1 = nn.ReLU(inplace=True)
        self.e3_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.e3_bn2 = nn.BatchNorm2d(256)
        self.e3_relu2 = nn.ReLU(inplace=True)
        
        # e4
        self.e4_maxpool = nn.MaxPool2d(kernel_size=2)
        self.e4_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.e4_bn1 = nn.BatchNorm2d(512)
        self.e4_relu1 = nn.ReLU(inplace=True)
        self.e4_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.e4_bn2 = nn.BatchNorm2d(512)
        self.e4_relu2 = nn.ReLU(inplace=True)
        
        # e5
        self.e5_maxpool = nn.MaxPool2d(kernel_size=2)
        self.e5_conv1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.e5_bn1 = nn.BatchNorm2d(1024)
        self.e5_relu1 = nn.ReLU(inplace=True)
        self.e5_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.e5_bn2 = nn.BatchNorm2d(1024)
        self.e5_relu2 = nn.ReLU(inplace=True)

        
        # Decoder
        
        # e1 -> d4
        self.d4_e1_maxpool = nn.MaxPool2d(kernel_size=8, stride=8, ceil_mode=True)
        self.d4_e1_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.d4_e1_bn = nn.BatchNorm2d(64)
        self.d4_e1_relu = nn.ReLU(inplace=True)
        # e2 -> d4
        self.d4_e2_maxpool = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.d4_e2_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.d4_e2_bn = nn.BatchNorm2d(64)
        self.d4_e2_relu = nn.ReLU(inplace=True)
        # e3 -> d4
        self.d4_e3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.d4_e3_conv = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.d4_e3_bn = nn.BatchNorm2d(64)
        self.d4_e3_relu = nn.ReLU(inplace=True)        
        # e4 -> d4
        self.d4_e4_conv = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)
        self.d4_e4_bn = nn.BatchNorm2d(64)
        self.d4_e4_relu = nn.ReLU(inplace=True)  
        # e5 -> d4
        self.d4_e5_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d4_e5_conv = nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1)
        self.d4_e5_bn = nn.BatchNorm2d(64)
        self.d4_e5_relu = nn.ReLU(inplace=True)  
        # d4
        self.d4_conv = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)
        self.d4_bn = nn.BatchNorm2d(320)
        self.d4_relu = nn.ReLU(inplace=True)  
        
        
        # e1 -> d3
        self.d3_e1_maxpool = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.d3_e1_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.d3_e1_bn = nn.BatchNorm2d(64)
        self.d3_e1_relu = nn.ReLU(inplace=True)
        # e2 -> d3
        self.d3_e2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.d3_e2_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.d3_e2_bn = nn.BatchNorm2d(64)
        self.d3_e2_relu = nn.ReLU(inplace=True)
        # e3 -> d3
        self.d3_e3_conv = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.d3_e3_bn = nn.BatchNorm2d(64)
        self.d3_e3_relu = nn.ReLU(inplace=True)  
        # d4 -> d3
        self.d3_d4_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d3_d4_conv = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.d3_d4_bn = nn.BatchNorm2d(64)
        self.d3_d4_relu = nn.ReLU(inplace=True)  
        # e5 -> d3
        self.d3_e5_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.d3_e5_conv = nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1)
        self.d3_e5_bn = nn.BatchNorm2d(64)
        self.d3_e5_relu = nn.ReLU(inplace=True)  
        # d3
        self.d3_conv = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)
        self.d3_bn = nn.BatchNorm2d(320)
        self.d3_relu = nn.ReLU(inplace=True) 
        
        
        # e1 -> d2
        self.d2_e1_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.d2_e1_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.d2_e1_bn = nn.BatchNorm2d(64)
        self.d2_e1_relu = nn.ReLU(inplace=True)
        # e2 -> d2
        self.d2_e2_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.d2_e2_bn = nn.BatchNorm2d(64)
        self.d2_e2_relu = nn.ReLU(inplace=True)  
        # d3 -> d2
        self.d2_d3_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d2_d3_conv = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.d2_d3_bn = nn.BatchNorm2d(64)
        self.d2_d3_relu = nn.ReLU(inplace=True) 
        # d4 -> d2
        self.d2_d4_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.d2_d4_conv = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.d2_d4_bn = nn.BatchNorm2d(64)
        self.d2_d4_relu = nn.ReLU(inplace=True) 
        # e5 -> d2
        self.d2_e5_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.d2_e5_conv = nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1)
        self.d2_e5_bn = nn.BatchNorm2d(64)
        self.d2_e5_relu = nn.ReLU(inplace=True) 
        # d2
        self.d2_conv = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)
        self.d2_bn = nn.BatchNorm2d(320)
        self.d2_relu = nn.ReLU(inplace=True) 
        
        
        # e1 -> d1
        self.d1_e1_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.d1_e1_bn = nn.BatchNorm2d(64)
        self.d1_e1_relu = nn.ReLU(inplace=True) 
        # d2 -> d1
        self.d1_d2_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d1_d2_conv = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.d1_d2_bn = nn.BatchNorm2d(64)
        self.d1_d2_relu = nn.ReLU(inplace=True) 
        # d3 -> d1
        self.d1_d3_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.d1_d3_conv = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.d1_d3_bn = nn.BatchNorm2d(64)
        self.d1_d3_relu = nn.ReLU(inplace=True) 
        # d4 -> d1
        self.d1_d4_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.d1_d4_conv = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.d1_d4_bn = nn.BatchNorm2d(64)
        self.d1_d4_relu = nn.ReLU(inplace=True) 
        # e5 -> d1
        self.d1_e5_upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.d1_e5_conv = nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1)
        self.d1_e5_bn = nn.BatchNorm2d(64)
        self.d1_e5_relu = nn.ReLU(inplace=True) 
        # d1
        self.d1_conv = nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1)
        self.d1_bn = nn.BatchNorm2d(320)
        self.d1_relu = nn.ReLU(inplace=True) 
        
        
        # Output
        
        self.output = nn.Conv2d(320, self.out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        
        e1 = self.e1_relu2(self.e1_bn2(self.e1_conv2(self.e1_relu1(self.e1_bn1(self.e1_conv1(x))))))
        
        e2 = self.e2_relu2(self.e2_bn2(self.e2_conv2(self.e2_relu1(self.e2_bn1(self.e2_conv1(self.e2_maxpool(e1)))))))
        
        e3 = self.e3_relu2(self.e3_bn2(self.e3_conv2(self.e3_relu1(self.e3_bn1(self.e3_conv1(self.e3_maxpool(e2)))))))
        
        e4 = self.e4_relu2(self.e4_bn2(self.e4_conv2(self.e4_relu1(self.e4_bn1(self.e4_conv1(self.e4_maxpool(e3)))))))
        
        e5 = self.e5_relu2(self.e5_bn2(self.e5_conv2(self.e5_relu1(self.e5_bn1(self.e5_conv1(self.e5_maxpool(e4)))))))
        
        d4_e1 = self.d4_e1_relu(self.d4_e1_bn(self.d4_e1_conv(self.d4_e1_maxpool(e1))))
        d4_e2 = self.d4_e2_relu(self.d4_e2_bn(self.d4_e2_conv(self.d4_e2_maxpool(e2))))
        d4_e3 = self.d4_e3_relu(self.d4_e3_bn(self.d4_e3_conv(self.d4_e3_maxpool(e3))))
        d4_e4 = self.d4_e4_relu(self.d4_e4_bn(self.d4_e4_conv(e4)))
        d4_e5 = self.d4_e5_relu(self.d4_e5_bn(self.d4_e5_conv(self.d4_e5_upsample(e5))))
        d4 = self.d4_relu(self.d4_bn(self.d4_conv(torch.cat((d4_e1, d4_e2, d4_e3, d4_e4, d4_e5), 1))))
        
        d3_e1 = self.d3_e1_relu(self.d3_e1_bn(self.d3_e1_conv(self.d3_e1_maxpool(e1))))
        d3_e2 = self.d3_e2_relu(self.d3_e2_bn(self.d3_e2_conv(self.d3_e2_maxpool(e2))))
        d3_e3 = self.d3_e3_relu(self.d3_e3_bn(self.d3_e3_conv(e3)))
        d3_d4 = self.d3_d4_relu(self.d3_d4_bn(self.d3_d4_conv(self.d3_d4_upsample(d4))))
        d3_e5 = self.d3_e5_relu(self.d3_e5_bn(self.d3_e5_conv(self.d3_e5_upsample(e5))))
        d3 = self.d3_relu(self.d3_bn(self.d3_conv(torch.cat((d3_e1, d3_e2, d3_e3, d3_d4, d3_e5), 1))))
        
        d2_e1 = self.d2_e1_relu(self.d2_e1_bn(self.d2_e1_conv(self.d2_e1_maxpool(e1))))
        d2_e2 = self.d2_e2_relu(self.d2_e2_bn(self.d2_e2_conv(e2)))
        d2_d3 = self.d2_d3_relu(self.d2_d3_bn(self.d2_d3_conv(self.d2_d3_upsample(d3))))
        d2_d4 = self.d2_d4_relu(self.d2_d4_bn(self.d2_d4_conv(self.d2_d4_upsample(d4))))
        d2_e5 = self.d2_e5_relu(self.d2_e5_bn(self.d2_e5_conv(self.d2_e5_upsample(e5))))
        d2 = self.d2_relu(self.d2_bn(self.d2_conv(torch.cat((d2_e1, d2_e2, d2_d3, d2_d4, d2_e5), 1))))
        
        d1_e1 = self.d1_e1_relu(self.d1_e1_bn(self.d1_e1_conv(e1)))
        d1_d2 = self.d1_d2_relu(self.d1_d2_bn(self.d1_d2_conv(self.d1_d2_upsample(d2))))
        d1_d3 = self.d1_d3_relu(self.d1_d3_bn(self.d1_d3_conv(self.d1_d3_upsample(d3))))
        d1_d4 = self.d1_d4_relu(self.d1_d4_bn(self.d1_d4_conv(self.d1_d4_upsample(d4))))
        d1_e5 = self.d1_e5_relu(self.d1_e5_bn(self.d1_e5_conv(self.d1_e5_upsample(e5))))
        d1 = self.d1_relu(self.d1_bn(self.d1_conv(torch.cat((d1_e1, d1_d2, d1_d3, d1_d4, d1_e5), 1))))
        
        output = self.output(d1)
        
        return torch.sigmoid(output)
