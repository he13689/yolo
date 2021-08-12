import os

import torch
import torch.nn as nn

from models.basemodel.d19 import CSPDarknetV5
from config import *
from utils.moduleUtils import CSPModule, Conv


class YOLOv5(nn.Module):
    # 复现v5， 只提取内核
    '''
    nc=80  分类有多少种
     e1=0.33, e2=0.5 缩放比例，方便控制网络的参数量
    '''

    def __init__(self, nc=80, e1=0.33, e2=0.5, pretrained=False):
        super(YOLOv5, self).__init__()
        self.base = CSPDarknetV5(e1, e2)
        if pretrained:
            path_to_dir = os.path.dirname(os.path.abspath(__file__))
            print('Loading the darknet')
            self.base.load_state_dict(torch.load(path_to_dir))

        self.neck_small = nn.Sequential(
            CSPModule(1024, 1024, n=3, e=[e1, e2]),
            Conv(round(1024 * e2), round(512 * e2), k=1, s=1, p=0)
        )

        self.neck_medium = nn.Sequential(
            CSPModule(1024, 512, n=3, e=[e1, e2]),
            Conv(round(512 * e2), round(256 * e2), 1, s=1, p=0)
        )

        self.tie_large = nn.Sequential(
            CSPModule(512, 256, n=3, e=[e1, e2])
        )

        self.tie_medium = nn.Sequential(
            CSPModule(512, 512, n=3, e=[e1, e2])
        )

        self.tie_small = nn.Sequential(
            CSPModule(1024, 1024, n=3, e=[e1, e2])
        )

        self.pan_medium = nn.Sequential(
            Conv(round(256 * e2), round(256 * e2), k=3, s=2, p=1)
        )

        self.pan_small = nn.Sequential(
            Conv(round(512 * e2), round(512 * e2), k=3, s=2, p=1)
        )

        # FPN：2次上采样 自顶而下 完成语义信息增强
        self.upsample_medium = nn.Upsample(scale_factor=2)
        self.upsample_large = nn.Upsample(scale_factor=2)

        #  detection
        self.det_large = nn.Conv2d(round(e2 * 256), 3 * (5 + nc), 1, 1, 0)
        self.det_medium = nn.Conv2d(round(512 * e2), 3 * (5 + nc), 1, 1, 0)
        self.det_small = nn.Conv2d(round(1024 * e2), 3 * (5 + nc), 1, 1, 0)

    def forward(self, x):
        hl, hm, hs = self.base(x)  # 从FPN中取出大中小三种特征
        ys = self.neck_small(hs)

        upm = self.upsample_medium(ys)
        mid = torch.cat([upm, hm], 1)  # 将small的结果和hm输出结合在一起
        ym = self.neck_medium(mid)

        upl = self.upsample_large(ym)
        large = torch.cat([upl, hl], 1)
        yl = self.tie_large(large)

        mid = torch.cat([self.pan_medium(yl), ym], dim=1)
        upm = self.tie_medium(mid)

        small = torch.cat([self.pan_small(upm), ys], 1)
        small = self.tie_small(small)

        ys = self.det_small(small)
        ym = self.det_medium(upm)
        yl = self.det_large(yl)

        return ys, ym, yl


if __name__ == '__main__':
    hyperparameter = {
        #            gd    gw
        'yolov5s': [0.33, 0.50],
        'yolov5m': [0.67, 0.75],
        'yolov5l': [1.00, 1.00],
        'yolov5x': [1.33, 1.25]
    }

    size = hyperparameter['yolov5s']
    net = YOLOv5(nc=20, e1=size[0], e2=size[1])
    print(net)
    img = torch.randn(1, 3, 512, 512)
    y = net(img)
    #  输出 torch.Size([1, 75, 16, 16]) torch.Size([1, 75, 32, 32]) torch.Size([1, 75, 64, 64])
    # 差距是 2^5=32  2^4=16  2^3=8   对应它们所做的 步长为2卷积个数
    print(y[0].size(), y[1].size(), y[2].size())
