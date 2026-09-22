"""WiSPPN CSI→PAM 회귀 네트워크 — 비공식 수정 구현 (unofficial modified implementation).

출처: https://github.com/geekfeiw/WiSPPN 의 models/wisppn_resnet.py 를 기반으로 수정.
논문: Fei Wang, Stanislav Panev, Ziyi Dai, Jinsong Han, Dong Huang,
      "Can WiFi Estimate Person Pose?", arXiv:1904.00277 (2019).
원 저장소에는 라이선스 파일이 없으므로 이 파일의 원본 부분 저작권은 원저자에게 있으며,
본 저장소는 출처 표기와 함께 사용한다 (README의 출처/라이선스 절 참조).
참고로 원본의 ResidualBlock·make_layer 골격 자체는 yunjey/pytorch-tutorial(MIT) 계열의
표준 ResNet 구현이다.

수정점: ① stem 280→150 정식 사용(원본은 conv1 미호출 버그) ② decode 최종 2ch→3ch(x,y,ĉ)
③ interpolate scale_factor=48 → size=(144,144) 명시(동등) ④ vector_head 변형.
"""
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class WiSPPN(nn.Module):
    """(B,in_ch,3,3) → (B,3,18,18) PAM(x,y,ĉ); vector_head=True → (B,3,18) 대각 직접."""

    def __init__(self, in_ch=280, width=150, layers=(2, 2, 2, 2), n_joints=18,
                 vector_head=False):
        super().__init__()
        self.n_joints = n_joints
        self.vector_head = vector_head
        self.stem = nn.Sequential(conv3x3(in_ch, width),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.in_channels = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width, layers[1], 2)
        self.layer3 = self._make_layer(width * 2, layers[2], 2)
        self.layer4 = self._make_layer(width * 2, layers[3], 2)
        if vector_head:
            self.head = nn.Sequential(nn.Linear(width * 2, 64), nn.ReLU(inplace=True),
                                      nn.Linear(64, 3 * n_joints))
        else:
            self.decode = nn.Sequential(            # 원본 decode + 최종 3ch(bias=False 원본 유지)
                conv3x3(width * 2, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 1, bias=False))

    def _make_layer(self, cout, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != cout:
            downsample = nn.Sequential(conv3x3(self.in_channels, cout, stride),
                                       nn.BatchNorm2d(cout))
        blocks_ = [ResidualBlock(self.in_channels, cout, stride, downsample)]
        self.in_channels = cout
        blocks_ += [ResidualBlock(cout, cout) for _ in range(blocks - 1)]
        return nn.Sequential(*blocks_)

    def forward(self, x):
        size = self.n_joints * 8                     # 144 — stride 2 ×3 후 18×18
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        if self.vector_head:
            return self.head(x.mean(dim=(2, 3))).view(-1, 3, self.n_joints)
        return self.decode(x)
