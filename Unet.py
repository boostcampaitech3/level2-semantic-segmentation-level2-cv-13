import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, num_classes=11):
        super(UNet, self).__init__()
        def convblock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU())

            return nn.Sequential(*layers)

        # encoder 1
        self.encoder1_1 = convblock(3, 64)
        self.encoder1_2 = nn.MaxPool2d(2, 2)

        # encoder 2
        self.encoder2_1 = convblock(64, 128)
        self.encoder2_2 = nn.MaxPool2d(2, 2)

        # encoder 3
        self.encoder3_1 = convblock(128, 256)
        self.encoder3_2 = nn.MaxPool2d(2, 2)

        # encoder 4
        self.encoder4_1 = convblock(256, 512)
        self.encoder4_2 = nn.MaxPool2d(2, 2)

        # encoder 5
        self.encoder5_1 = convblock(512, 1024)

        # decoder 1
        self.decoder1_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.decoder1_2 = convblock(1024, 512)

        # decoder 2
        self.decoder2_1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.decoder2_2 = convblock(512, 256)

        # decoder 3
        self.decoder3_1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.decoder3_2 = convblock(256, 128)

        # decoder 4
        self.decoder4_1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.decoder4_2 = convblock(128, 64)
        self.segmap = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # encoder
        encoder1 = self.encoder1_1(x)
        encoder1_2 = self.encoder1_2(encoder1)

        encoder2 = self.encoder2_1(encoder1_2)
        encoder2_2 = self.encoder2_2(encoder2)

        encoder3 = self.encoder3_1(encoder2_2)
        encoder3_2 = self.encoder3_2(encoder3)

        encoder4 = self.encoder4_1(encoder3_2)
        encoder4_2 = self.encoder4_2(encoder4)

        encoder5 = self.encoder5_1(encoder4_2)

        # decoder
        decoder1_1 = self.decoder1_1(encoder5)
        decoder1_2 = torch.cat((decoder1_1, encoder4), dim=1)
        decoder1_3 = self.decoder1_2(decoder1_2)

        decoder2_1 = self.decoder2_1(decoder1_3)
        decoder2_2 = torch.cat((decoder2_1, encoder3), dim=1)
        decoder2_3 = self.decoder2_2(decoder2_2)

        decoder3_1 = self.decoder3_1(decoder2_3)
        decoder3_2 = torch.cat((decoder3_1, encoder2), dim=1)
        decoder3_3 = self.decoder3_2(decoder3_2)

        decoder4_1 = self.decoder4_1(decoder3_3)
        decoder4_2 = torch.cat((decoder4_1, encoder1), dim=1)
        decoder4_3 = self.decoder4_2(decoder4_2)

        output = self.segmap(decoder4_3)

        return output


# model = UNet(num_classes=11)
# x = torch.randn([1, 3, 512, 512])
# print("input shape : ", x.shape)
# out = model(x).to(device)
# print("output shape : ", out.size())

# model = model.to(device)
