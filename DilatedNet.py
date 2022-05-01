from turtle import forward
from cv2 import dilate
import torch
import torch.nn as nn

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, dilate_rate, kernel_size) -> None:
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilate_rate, kernel_size=kernel_size, padding=dilate_rate, stride=1),
                                    nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class FrontEnd(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
                                ConvRelu(in_channels=3, out_channels=64, dilate_rate=1, kernel_size=3),
                                ConvRelu(in_channels=64, out_channels=64, dilate_rate=1, kernel_size=3),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.conv2 = nn.Sequential(
                                ConvRelu(in_channels=64, out_channels=128, dilate_rate=1, kernel_size=3),
                                ConvRelu(in_channels=128, out_channels=128, dilate_rate=1, kernel_size=3),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.conv3 = nn.Sequential(
                                ConvRelu(in_channels=128, out_channels=256, dilate_rate=1, kernel_size=3),
                                ConvRelu(in_channels=256, out_channels=256, dilate_rate=1, kernel_size=3),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.conv4 = nn.Sequential(
                                ConvRelu(in_channels=256, out_channels=512, dilate_rate=1, kernel_size=3),
                                ConvRelu(in_channels=512, out_channels=512, dilate_rate=1, kernel_size=3),
                                ConvRelu(in_channels=512, out_channels=512, dilate_rate=1, kernel_size=3),
                                    )
        self.conv5 = nn.Sequential(
                                ConvRelu(in_channels=512, out_channels=512, dilate_rate=2, kernel_size=3),
                                ConvRelu(in_channels=512, out_channels=512, dilate_rate=2, kernel_size=3),
                                ConvRelu(in_channels=512, out_channels=512, dilate_rate=2, kernel_size=3),
                                    )

        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=7, dilation=4, padding=12), 
                                        nn.ReLU(),
                                        nn.Dropout2d(0.5),
                                        nn.Conv2d(4096, 4096, kernel_size=1),
                                        nn.ReLU(),
                                        nn.Dropout2d(0.5),
                                        nn.Conv2d(4096, num_classes, kernel_size=1)
                                        )

    def forward(self, x): 
        out = self.classifier(x)
        return out




class ContextModule(nn.Module):
    def __init__(self, num_classes = 11, option ="Basic") :
        super().__init__()

        if option == "Large":
            channels = [1, 2, 2, 4, 8, 16, 32, 32, 1]
        elif option == "Basic":
            channels = [1 for _ in range(9)]
        else:
            # error
            print("error")

        self.layer1 = ConvRelu(in_channels=num_classes*channels[0], out_channels=num_classes*channels[1], dilate_rate=1, kernel_size=3)
        self.layer2 = ConvRelu(in_channels=num_classes*channels[1], out_channels=num_classes*channels[2], dilate_rate=1, kernel_size=3)
        self.layer3 = ConvRelu(in_channels=num_classes*channels[2], out_channels=num_classes*channels[3], dilate_rate=2, kernel_size=3)
        self.layer4 = ConvRelu(in_channels=num_classes*channels[3], out_channels=num_classes*channels[4], dilate_rate=4, kernel_size=3)
        self.layer5 = ConvRelu(in_channels=num_classes*channels[4], out_channels=num_classes*channels[5], dilate_rate=8, kernel_size=3)
        self.layer6 = ConvRelu(in_channels=num_classes*channels[5], out_channels=num_classes*channels[6], dilate_rate=16, kernel_size=3)
        self.layer7 = ConvRelu(in_channels=num_classes*channels[6], out_channels=num_classes*channels[7], dilate_rate=1, kernel_size=3)
        self.layer8 = nn.Conv2d(in_channels=num_classes*channels[7], out_channels=num_classes*channels[8], dilation=1, kernel_size=1)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7, self.layer8]


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class DeConv(nn.Module):
    def __init__(self, num_classes):
        
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8, padding=4)
        
    def forward(self, x):
        x = self.deconv(x)
        return x



class DilatedNet(nn.Module):

    def __init__(self, num_classes, option) -> None:
        super().__init__()
        self.frontEnd = FrontEnd()
        self.classifier = Classifier(num_classes = num_classes)
        self.contextModule = ContextModule(num_classes=num_classes, option=option) 
        self.deconv = DeConv(num_classes=num_classes)

    def forward(self, x):
        x = self.frontEnd(x)
        x = self.classifier(x)
        x = self.contextModule(x)
        x = self.deconv(x)
        return x



model = DilatedNet(num_classes=11, option="Basic")
x = torch.randn([4, 3, 512, 512])
print("input shape : ", x.shape)
out = model(x)
print("output shape : ", out.size())