import torch
import torch.nn as nn
import torchvision
import timm


class FCN(nn.Module):
    """Fully Convolutional Network - segmentation
        
    FCN 모델 구현. timm 모델을 백본으로 사용하며 오리지널 FCN과 구현 일부가 다르다.
    (paper: https://arxiv.org/abs/1411.4038)
    
    구현 차이점:
        * VGG16외의 다양한 백본 선택 가능.
            + fully-connected 레이어를 대체한 conv 레이어 중 첫번째 레이어의 커널을 7x7에서 1x1으로 수정함.
            + classifier를 대체하는 conv 레이어들이 classifier의 가중치를 이어받지않고 새롭게 초기화됨.
        * score conv layer가 zero initialization 대신 he initialization 적용.
        * deconv 레이어에 bilinear 대신 he initialization 적용.
        * 스킵커넥션은 다운샘플링 레이어의 출력이 아닌 다운샘플링 레이어의 입력을 사용함.
    
    Attributes:
        num_classes (int): 배경을 포함한 클래스 수
        backbone (nn.Module): 백본 모델
    """
    def __init__(self,
                 num_classes_wo_bg: int = 10,
                 backbone: str = 'vgg16',
                 fcn_name: str = '32s',
                 pretrained_backbone: bool = True,
                 use_bn: bool = False):
        """FCN 객체 생성
        
        Args:
            num_classes_wo_bg (int): 배경을 제외한 클래스 수
            backbone (str): 백본으로 사용하려는 timm 모델 이름
            fcn_name (str): FCN 모델 이름 / ( 32s | 16s | 8s )
            pretrained_backbone (bool): 백본의 기학습 가중치 로드 여부
            use_bn (bool): 백본 이후에 배치정규화 사용 여부
        """
        super().__init__()
        
        if fcn_name not in ['8s', '16s', '32s']:
            raise ValueError(f'지원되는 FCN 구조 - [8s, 16s, 32s] / 입력: {fcn_name}')
        self.fcn_name = fcn_name
        
        num_classes = num_classes_wo_bg + 1
        
        # backbone
        self.backbone = timm.create_model(backbone, features_only=True, pretrained=pretrained_backbone)
        feats_info = self.backbone.feature_info.channels()
        
        # FCN32s
        def conv_1x1(in_ch, out_ch, k, s, use_bn):
            layers = [
                nn.Conv2d(in_ch, out_ch, k, s),
                nn.ReLU(True),
                nn.Dropout2d(),
            ]
            if use_bn:
                layers.insert(1, nn.BatchNorm2d(out_ch))
            return nn.Sequential(*layers)
        
        self.conv_32s_0 = conv_1x1(feats_info[-1], 4096, 1, 1, use_bn)  # 7x7 커널을 1x1으로 교체
        self.conv_32s_1 = conv_1x1(4096, 4096, 1, 1, use_bn)
        self.conv_32s_final = nn.Conv2d(4096, num_classes, 1, 1)
        
        # FCN16s, FCN8s
        k_32s = 64 if fcn_name == '32s' else 4
        self.deconv_32s = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=k_32s, stride=k_32s//2, padding=k_32s//4)
        
        if fcn_name in ['8s', '16s']:
            self.conv_16s = nn.Conv2d(feats_info[-2], num_classes, 1, 1)
            k_16s = 32 if fcn_name == '16s' else 4
            self.deconv_16s = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=k_16s, stride=k_16s//2, padding=k_16s//4)
            
            if fcn_name == '8s':
                self.conv_8s = nn.Conv2d(feats_info[-3], num_classes, 1, 1)
                self.deconv_8s = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)
                
        
    def forward(self, x):
        feats = self.backbone(x)
        x = self.conv_32s_0(feats[-1])
        x = self.conv_32s_1(x)
        x = self.conv_32s_final(x)
        x = self.deconv_32s(x)
        
        if self.fcn_name != '32s':
            skip_16s = self.conv_16s(feats[-2])
            x = x + skip_16s
            x = self.deconv_16s(x)
        
        if self.fcn_name == '8s':
            skip_8s = self.conv_8s(feats[-3])
            x = x + skip_8s
            x = self.deconv_8s(x)
        return x