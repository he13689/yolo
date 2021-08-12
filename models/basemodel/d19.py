# darknet 19

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
from utils.moduleUtils import Conv, resblock, CSPStage, Focus, CSPModule, SPP


# 算上conv7一共有19层conv，x1,x2,x3代表不同层级的特征图
# 包括19个卷积层和5个maxpooling层， Darknet-19最终采用global avgpooling做预测，并且在 3X3 卷积之间使用 1X1 卷积来压缩特征图channles以降低模型计算量和参数。
# Darknet-19每个卷积层后面同样使用了batch norm层以加快收敛速度，降低模型过拟合。
# 这是一个较大的完整模型，用于处理
class Dark19(nn.Module):
    def __init__(self, num_classes=1000):
        super(Dark19, self).__init__()
        self.conv1 = nn.Sequential(
            Conv(3, 32, k=3, p=1),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            Conv(32, 64, k=3, p=1),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            Conv(64, 128, k=3, p=1),
            Conv(128, 64, k=1),
            Conv(64, 128, k=3, p=1),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1),
            Conv(128, 256, k=3, p=1),
        )

        self.maxpool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1),
        )

        self.maxpool5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Sequential(
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
        )

        # self.conv_7 = nn.Conv2d(1024, 1000, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.conv4(x)
        x2 = self.conv5(self.maxpool4(x1))
        x3 = self.conv6(self.maxpool5(x2))

        # x = self.conv_7(C_6)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # return x

        return x1, x2, x3


# darknet 53 加入了resblock，共有53层卷积
class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """

    def __init__(self, num_classes=1000):
        super(DarkNet_53, self).__init__()
        # stride = 2 对应yolov3.jpg 中 res1
        self.layer_1 = nn.Sequential(
            Conv(3, 32, 3, p=1),
            Conv(32, 64, 3, p=1, s=2),
            resblock(64, nblocks=1)
        )
        # stride = 4 res2
        self.layer_2 = nn.Sequential(
            Conv(64, 128, 3, p=1, s=2),
            resblock(128, nblocks=2)
        )
        # stride = 8 res8
        self.layer_3 = nn.Sequential(
            Conv(128, 256, 3, p=1, s=2),
            resblock(256, nblocks=8)
        )
        # stride = 16 res8
        self.layer_4 = nn.Sequential(
            Conv(256, 512, 3, p=1, s=2),
            resblock(512, nblocks=8)
        )
        # stride = 32 res4
        self.layer_5 = nn.Sequential(
            Conv(512, 1024, 3, p=1, s=2),
            resblock(1024, nblocks=4)
        )

    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        C_3 = self.layer_3(x)
        C_4 = self.layer_4(C_3)
        C_5 = self.layer_5(C_4)
        return C_3, C_4, C_5


# 缩小版darknet，c更小 适用于测试运行， 并且没有用到pool，只涉及卷积
class DarkNet_Tiny(nn.Module):
    def __init__(self, num_classes=1000):
        super(DarkNet_Tiny, self).__init__()

        self.conv_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1),
            Conv(32, 32, k=3, p=1, s=2)
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv(32, 64, k=3, p=1),
            Conv(64, 64, k=3, p=1, s=2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv(64, 128, k=3, p=1),
            Conv(128, 128, k=3, p=1, s=2),
        )

        # output : stride = 16, c = 256
        self.conv_4 = nn.Sequential(
            Conv(128, 256, k=3, p=1),
            Conv(256, 256, k=3, p=1, s=2),
        )

        # output : stride = 32, c = 512
        self.conv_5 = nn.Sequential(
            Conv(256, 512, k=3, p=1),
            Conv(512, 512, k=3, p=1, s=2),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        C_3 = self.conv_3(x)
        C_4 = self.conv_4(C_3)
        C_5 = self.conv_5(C_4)
        return C_3, C_4, C_5


# 更加轻量级的Darknet， 比tiny还要小 适用于移动设备， 只有7个conv
class DarkNet_Light(nn.Module):
    def __init__(self, num_classes=1000):
        super(DarkNet_Light, self).__init__()
        # backbone network : DarkNet_Light
        self.conv_1 = Conv(3, 16, k=3, p=1)
        self.maxpool_1 = nn.MaxPool2d((2, 2), 2)  # stride = 2

        self.conv_2 = Conv(16, 32, k=3, p=1)
        self.maxpool_2 = nn.MaxPool2d((2, 2), 2)  # stride = 4

        self.conv_3 = Conv(32, 64, k=3, p=1)
        self.maxpool_3 = nn.MaxPool2d((2, 2), 2)  # stride = 8

        self.conv_4 = Conv(64, 128, k=3, p=1)
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)  # stride = 16

        self.conv_5 = Conv(128, 256, k=3, p=1)
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)  # stride = 32

        self.conv_6 = Conv(256, 512, k=3, p=1)
        self.maxpool_6 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d((2, 2), 1)  # stride = 32
        )

        self.conv_7 = Conv(512, 1024, k=3, p=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.maxpool_3(x)
        x = self.conv_4(x)
        x = self.maxpool_4(x)
        C_4 = self.conv_5(x)  # stride = 16
        x = self.maxpool_5(C_4)
        x = self.conv_6(x)
        x = self.maxpool_6(x)
        C_5 = self.conv_7(x)  # stride = 32

        return C_4, C_5


class CSPDarknet(nn.Module):
    '''
    Cross-stage partial connections 详情可见 Darknet53结构图
    CSPDarknet53是在Darknet53的每个大残差块上加上CSP，

    在 Darknet53分块1加上CSP后的结果 图中可见

    Darknet53分块1加上CSP后的结果，对应layer 0~layer 10。其中，layer [0, 1, 5, 6, 7]与分块1完全一样，而 layer [2, 4, 8, 9, 10]属于CSP部分。
    Darknet53分块2加上CSP后的结果，对应layer 11~layer 23。其中，layer [11, 15~20]对应分块2（注意:残差单元中的3 × 3 卷积核的深度改变了，由Darknet53分块2中的128改为64，请看layer 16 和 layer 19），其余 layer属于CSP部分。
    Darknet53分块3加上CSP后的结果，对应layer 24~layer 54。其中，layer [24, 27~51]对应分块3（注意:残差单元中的3 × 3 卷积核的深度改变了，由Darknet53分块3中的256改为128，请看layer 29等），其余 layer属于CSP部分。
    Darknet53分块4加上CSP后的结果，对应layer 55~layer 85。其中，layer [55, 58~82]对应分块4（注意:残差单元中的3 × 3 卷积核的深度改变了，由Darknet53分块4中的512改为256，请看layer 60等），其余 layer属于CSP部分。
    Darknet53分块5加上CSP后的结果，对应layer 86~layer 104。其中，layer [86, 89~101]对应分块5（注意:残差单元中的3 × 3 卷积核的深度改变了，由Darknet53分块5中的1024改为512，请看layer 91等），其余 layer属于CSP部分。

    在目标检测领域的精度来说，CSPDarknet53是要强于 CSPResNext50，这也告诉了我们，在图像分类上任务表现好的模型，不一定很适用于目标检测

    '''

    def __init__(self, num_classes=1000):
        super(CSPDarknet, self).__init__()
        # 假设输入图片大小是608 x 608 x 3
        # blocks : [1 2 8 8 4]
        self.layer1 = nn.Sequential(
            Conv(3, 32, k=3, p=1),  # 3->32
            Conv(32, 64, k=3, p=1, s=2),  # 降低特征图宽度和高度 size // 2， 特征图尺寸 304 x 304 x 64
            CSPStage(c=64, n=1)
        )
        self.layer2 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2),
            CSPStage(128, n=2)  # P2/4
        )
        self.layer3 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2),
            CSPStage(256, n=8)  # P3/8
        )
        self.layer4 = nn.Sequential(
            Conv(256, 512, k=3, p=1, s=2),
            CSPStage(512, n=8)  # P4/16
        )
        self.layer5 = nn.Sequential(
            Conv(512, 1024, k=3, p=1, s=2),
            CSPStage(1024, n=4)  # P5/32
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        c3 = self.layer3(y)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        # 返回三个层次不同感受野下得到的特征
        return c3, c4, c5


class CSPDarknetV5(nn.Module):
    def __init__(self, e1, e2):
        super(CSPDarknetV5, self).__init__()
        self.big = nn.Sequential(
            Focus(3, 64, e=e2),
            Conv(round(64 * e2), round(128 * e2), k=3, s=2, p=1),
            CSPModule(128, 128, n=3, e=[e1, e2]),
            Conv(round(128 * e2), round(256 * e2), k=3, s=2, p=1),
            CSPModule(256, 256, n=9, e=[e1, e2]),
        )
        self.small = nn.Sequential(
            Conv(round(512 * e2), round(1024 * e2), k=3, s=2, p=1),
            Conv(round(1024 * e2), round(1024 * e2) // 2, k=1, s=1),  # cbl_before  未完成！！！！！！！！！
            SPP(),
            Conv(round(1024 * e2) // 2 * 4, round(1024 * e2), k=1, s=1),  # cbl_after
        )
        self.medium = nn.Sequential(
            Conv(round(256 * e2), round(512 * e2), k=3, s=2, p=1),
            CSPModule(512, 512, n=9, e=[e1, e2]),
        )

    def forward(self, x):
        hb = self.big(x)
        hm = self.medium(hb)
        hs = self.small(hm)
        # 返回三个不同规格的hidden vector
        return hb, hm, hs
