# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path
import torch

from yolov5_official.EMAModel import copy_attr
from yolov5_official.modules import *
import yaml
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


# 可以完全读取 v5s pt 的数据 效果很好
from yolov5_official.utils import fuse_conv_and_bn, AutoShape


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True, use_gpu=False):  # detection layer
        '''
        Detect 模块
        :param nc:  class 数目
        :param anchors:  anchors 的不同规格
        :param ch:  ch 有多个输入维度，表示多个不同代销输入的特征图
        :param inplace: 使用inplace 可以用于分割任务
        '''
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        # number of detection layers 和anchors数量有关 数量为3 因为对应不同特征图大小我们使用不同大小的anchor set， 不同set对应不同的detect layer
        self.nl = len(anchors)  # 一共有三组anchor， 每组anchor对应不同大小的特征图
        self.na = len(anchors[0]) // 2  # number of anchors =3 代表每个anchor set 有 三个规格
        self.grid = [torch.zeros(1)] * self.nl  # init grid [0,0,0]
        a = torch.Tensor(anchors).float().view(self.nl, -1, 2)  # (3,6) -> (3,3,2)  每个 大小的特征图对应一个anchor
        # shape(nl,na,2)  向模块添加持久缓冲区。  这通常用于注册不应被视为模型参数的缓冲区。 缓冲区可以使用给定的名称作为属性访问。 应该就是在内存中定义一个常量，同时，模型保存和加载的时候可以写入和读出
        self.register_buffer('anchors', a)
        # output conv 输出是self.no * self.na，表示每个anchor都有 no个输出
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace
        self.use_gpu = use_gpu

    def forward(self, x):
        '''
        torch.Size([1, 128, 32, 32])
        torch.Size([1, 256, 16, 16])
        torch.Size([1, 512, 8, 8])
        '''
        # print('detect 输入内容：')
        # for j in range(len(x)):
        #     print(x[j].shape)
        z = []
        for i in range(self.nl):  # 有几组anchor， 表示输入x是 几个不同大小的特征图
            # 对于训练过程， 就是简单地传入 样本 然后产生输出的一个过程
            x[i] = self.m[i](x[i])  # 将对应大小的输入传入对应的module list
            bs, _, ny, nx = x[i].shape
            # [bs, 3, outputs, ny, nx]   等于将x的第二个维度拆分为3*outputs num， 意义是对于batch中的每个elem
            # 都对应一个anchor set的结果，这个结果是 class， conf， pos 组成的
            # x中存储的是三个不同大小特征图 所对应的的 输出结果 [16, 3, 80, 80, 85] [16, 3, 40, 40, 85]
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # 训练阶段将跳过这里，直接将conv输出的结果经过变换后输出
            if not self.training:  # 如果不是训练阶段， 使用AWS 推测法   self.training 会随着model.train和model.eval变化， 输出结果也不同
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx=nx, ny=ny).to(x[i].device)

                y = x[i].sigmoid()  # 对先验框结果进行sigmoid激活函数
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))  # 最后的输出是 [bs, 25200, 85]
        # 如果是training， 那么返回 预测的框  如果是在val或者test阶段， model.eval()后training是false
        # 此时返回的结果是  三套anchor的先验框预测结果是x  z是对x的结果进行slice sigmoid之后的结果
        return x if self.training else (torch.cat(z, 1), x)  # 等于model最终输出

    def _make_grid(self, nx=20, ny=20, use_gpu=False):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    '''
    Focus结构

    Yolov5与Yolov4不同点在于，Yolov4中只有主干网络使用了CSP结构。 而Yolov5中设计了两种CSP结构，以Yolov5s网络为例，CSP1_X结构应用于Backbone主干网络，另一种CSP2_X结构则应用于Neck中。

    Yolov5现在的Neck和Yolov4中一样，都采用FPN+PAN的结构，但在Yolov5刚出来时，只使用了FPN结构，后面才增加了PAN结构，此外网络中其他部分也进行了调整。
    借鉴CSPnet设计的CSP2结构，加强网络特征融合的能力。

    Yolov5中采用其中的GIOU_Loss做Bounding box的损失函数。 而Yolov4中采用CIOU_Loss作为目标Bounding box的损失。

    在同样的参数情况下，将nms中IOU修改成DIOU_nms。对于一些遮挡重叠的目标，确实会有一些改进。

    网络使用CSP模块，先将基础层的特征映射划分为两部分，然后通过跨阶段层次结构将它们合并，在减少了计算量的同时可以保证准确率。
    增强CNN的学习能力，使得在轻量化的同时保持准确性，减少了梯度信息的重复
    降低计算瓶颈      降低内存成本

    Yolov4中使用的Dropblock，其实和常见网络中的Dropout功能类似，也是缓解过拟合的一种正则化方式。
    随机删除减少神经元的数量，使网络变得更简单
    Dropblock的研究者认为，卷积层对于这种随机丢弃并不敏感，因为卷积层通常是三层连用：卷积+激活+池化层，池化层本身就是对相邻单元起作用。而且即使随机丢弃，卷积层仍然可以从相邻的激活单元学习到相同的信息。
    Dropblock的研究者则干脆整个局部区域进行删减丢弃。这种方式其实是借鉴2017年的cutout数据增强的方式，cutout是将输入图像的部分区域清零，而Dropblock则是将Cutout应用到每一个特征图，而且并不是用固定的归零比率，而是在训练时以一个小的比率开始，随着训练过程线性的增加这个比率。

    优点一：Dropblock的效果优于Cutout
    优点二：Cutout只能作用于输入层，而Dropblock则是将Cutout应用到网络中的每一个特征图上
    优点三：Dropblock可以定制各种组合，在训练的不同阶段可以修改删减的概率，从空间层面和时间层面，和Cutout相比都有更精细的改进。
    '''

    def __init__(self, cfg='yolov5_official/yamls/yolov5s.yaml', ch=3, nc=None, anchors=None):
        '''
        :param cfg:  配置文件所在位置
        :param ch:  输入通道数
        :param nc:  class 数量
        :param anchors:  锚点数量

        model 结构
        0 (-1, 1, 'Focus', [64, 3])
        1 (-1, 1, 'Conv', [128, 3, 2])
        2 (-1, 3, 'C3', [128])
        3 (-1, 1, 'Conv', [256, 3, 2])
        4 (-1, 9, 'C3', [256])
        5 (-1, 1, 'Conv', [512, 3, 2])
        6 (-1, 9, 'C3', [512])
        7 (-1, 1, 'Conv', [1024, 3, 2])
        8 (-1, 1, 'SPP', [1024, [5, 9, 13]])
        9 (-1, 3, 'C3', [1024, False])
        10 (-1, 1, 'Conv', [512, 1, 1])
        11 (-1, 1, 'nn.Upsample', ['None', 2, 'nearest'])
        12 ([-1, 6], 1, 'Concat', [1])
        13 (-1, 3, 'C3', [512, False])
        14 (-1, 1, 'Conv', [256, 1, 1])
        15 (-1, 1, 'nn.Upsample', ['None', 2, 'nearest'])
        16 ([-1, 4], 1, 'Concat', [1])
        17 (-1, 3, 'C3', [256, False])
        18 (-1, 1, 'Conv', [256, 3, 2])
        19 ([-1, 14], 1, 'Concat', [1])
        20 (-1, 3, 'C3', [512, False])
        21 (-1, 1, 'Conv', [512, 3, 2])
        22 ([-1, 10], 1, 'Concat', [1])
        23 (-1, 3, 'C3', [1024, False])
        24 ([17, 20, 23], 1, 'Detect', ['nc', 'anchors'])
        '''

        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:
            self.yaml_file = Path(cfg).name
            # 从yaml中读取 model 结构
            with open(cfg) as file:
                self.yaml = yaml.safe_load(file)

        # 定义 model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            # 表示用形参中的参数代替 原始的 nc
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            # 用形参中的anchor代替原始的 anchors
            print("Overriding model.yaml anchors")
            self.yaml['anchors'] = round(anchors)

        self.model, self.save = parse_model(deepcopy(self.yaml), inch=[ch])  # 解析 model
        self.names = [str(i) for i in range(self.yaml['nc'])]  # name， 根据 class 数量，每个class对应一个i
        self.inplace = self.yaml.get('inplace', True)  # True - default -- 如果指定键的值不存在时，返回该默认值。

        m = self.model[-1]  # 取出 model 的最后一层

        if isinstance(m, Detect):  # 如果最后一层是detect的话
            stride = 256  # 计算 stride
            m.inplace = self.inplace
            #  m.stride 的大小通过stride / x.shape[-2] 来进行确定
            m.stride = torch.Tensor(
                [stride / x.shape[-2] for x in self.forward(torch.zeros(1, ch, stride, stride))])  # 计算出m 对应的stride
            m.anchors /= m.stride.view(-1, 1, 1)  # 计算anchors  需要用anchors除以stride
            check_anchor_order(m)  # 检查anchor的顺序是否正确
            self.stride = m.stride
            self._initialize_biases()  # 初始化detect biases

        initialize_weights(self)
        self.trainable = True

    def _initialize_biases(self, cf=None):
        # 初始化detect中conv的bias
        m = self.model[-1]
        for mod, s in zip(m.m, m.stride):  # stride和moduleList中的conv数量相同
            b = mod.bias.view(m.na, -1)  # 取出m.m中的bias，并且reshape

            # 重新对b进行计算
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mod.bias = nn.Parameter(b.view(-1), requires_grad=True)  # 使用计算后的b作为m中conv的参数

    def forward(self, x, augment=False):
        # 不论是训练还是测试，这里一定会用到detect
        if augment:
            return self.forward_augment(x)
        else:
            return self.forward_once(x)

    def forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        fi = [None, 3, None]
        y = []
        for si, fi in zip(s, fi):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # 调用forward once
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # 结果增强

    def forward_once(self, x):
        y = []  # outputs
        # 这个for的作用是，首先我们在self.save中预留设定了我们需要预留的结果
        # 当结果index满足save时，将中间结果x存在y中，当用到的时候m.f != -1，就把结果读取出来
        for m in self.model:  # 遍历模型中的所有module
            if m.f != -1:  # if not from previous layer 这就是说如果我们当前m的输入是从两个以上层的输出中得到的
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # 如果m的index是在save列表中的，此时我们就取出x放到y中
        '''
        torch.Size([1, 3, 32, 32, 85]) torch.Size([1, 3, 16, 16, 85]) torch.Size([1, 3, 8, 8, 85])
        '''
        # print('model 最终输出：',x[0].shape, x[1].shape, x[2].shape)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def autoshape(self):  # add AutoShape module
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

#
# class ModelWithAtten(nn.Module):
#     def __init__(self, cfg='yolov5_official/yamls/yolov5s.yaml', ch=3, nc=None, anchors=None):
#         super(ModelWithAtten, self).__init__()
#         self.yaml_file = Path(cfg).name
#         # 从yaml中读取 model 结构
#         with open(cfg) as f:
#             self.yaml = yaml.safe_load(f)
#
#         # 定义 model
#         ch = self.yaml['ch'] = self.yaml.get('ch', ch)
#         if nc and nc != self.yaml['nc']:
#             # 表示用形参中的参数代替 原始的 nc
#             print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
#             self.yaml['nc'] = nc
#         if anchors:
#             # 用形参中的anchor代替原始的 anchors
#             print("Overriding model.yaml anchors")
#             self.yaml['anchors'] = round(anchors)
#
#         self.model, self.save = parse_model(deepcopy(self.yaml), inch=[ch])  # 解析 model
#         self.names = [str(i) for i in range(self.yaml['nc'])]  # name， 根据 class 数量，每个class对应一个i
#         self.inplace = self.yaml.get('inplace', True)  # True - default -- 如果指定键的值不存在时，返回该默认值。
#
#         m = self.model[-1]  # 取出 model 的最后一层
#
#         # torch.Size([1, 128, 32, 32])
#         # torch.Size([1, 256, 16, 16])
#         # torch.Size([1, 512, 8, 8])
#         # 增加attention的方法之一， 其作用是在Basenet、PAN、FPN之后加入transformer block， 接在Detect之前，
#         self.transformBlock1 = TransformerBlock(128, 128, num_heads=16, num_layers=1)
#         self.transformBlock2 = TransformerBlock(256, 256, num_heads=16, num_layers=1)
#         self.transformBlock3 = TransformerBlock(512, 512, num_heads=16, num_layers=1)
#
#         if isinstance(m, Detect):  # 如果最后一层是detect的话
#             stride = 256  # 计算 stride
#             m.inplace = self.inplace
#             #  m.stride 的大小通过stride / x.shape[-2] 来进行确定
#             m.stride = torch.Tensor(
#                 [stride / x.shape[-2] for x in self.forward(torch.zeros(1, ch, stride, stride))])  # 计算出m 对应的stride
#             m.anchors /= m.stride.view(-1, 1, 1)  # 计算anchors  需要用anchors除以stride
#             check_anchor_order(m)  # 检查anchor的顺序是否正确
#             self.stride = m.stride
#             self._initialize_biases()  # 初始化detect biases
#
#         initialize_weights(self)
#         self.trainable = True
#
#     def _initialize_biases(self, cf=None):
#         # 初始化detect中conv的bias
#         m = self.model[-1]
#         for mod, s in zip(m.m, m.stride):  # stride和moduleList中的conv数量相同
#             b = mod.bias.view(m.na, -1)  # 取出m.m中的bias，并且reshape
#
#             # 重新对b进行计算
#             b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
#             b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
#             mod.bias = nn.Parameter(b.view(-1), requires_grad=True)  # 使用计算后的b作为m中conv的参数
#
#     def forward(self, x, augment=False):
#         # 不论是训练还是测试，这里一定会用到detect
#         if augment:
#             return self.forward_augment(x)
#         else:
#             return self.forward_once(x)
#
#     def forward_augment(self, x):
#         img_size = x.shape[-2:]  # height, width
#         s = [1, 0.83, 0.67]  # scales
#         f = [None, 3, None]  # flips (2-ud, 3-lr)
#         y = []  # outputs
#         for si, fi in zip(s, f):
#             xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
#             yi = self.forward_once(xi)[0]  # forward
#             # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
#             yi = self._descale_pred(yi, fi, si, img_size)
#             y.append(yi)
#         return torch.cat(y, 1), None  # augmented inference, train
#
#     def forward_once(self, x):
#         y = []  # outputs
#         # 这个for的作用是，首先我们在self.save中预留设定了我们需要预留的结果
#         # 当结果index满足save时，将中间结果x存在y中，当用到的时候m.f != -1，就把结果读取出来
#         for m in self.model:  # 遍历模型中的所有module
#             if m.f != -1:  # if not from previous layer 这就是说如果我们当前m的输入是从两个以上层的输出中得到的
#                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
#             if isinstance(m, Detect):
#                 # torch.Size([16, 128, 80, 80]) torch.Size([16, 256, 40, 40]) torch.Size([16, 512, 20, 20])
#                 x = [self.transformBlock1(x[0]), self.transformBlock2(x[1]), self.transformBlock3(x[2])]
#             x = m(x)  # run
#             y.append(x if m.i in self.save else None)  # 如果m的index是在save列表中的，此时我们就取出x放到y中
#         '''
#         torch.Size([1, 3, 32, 32, 85]) torch.Size([1, 3, 16, 16, 85]) torch.Size([1, 3, 8, 8, 85])
#         '''
#         # print('model 最终输出：',x[0].shape, x[1].shape, x[2].shape)
#         return x
#
#     def _descale_pred(self, p, flips, scale, img_size):
#         # de-scale predictions following augmented inference (inverse operation)
#         if self.inplace:
#             p[..., :4] /= scale  # de-scale
#             if flips == 2:
#                 p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
#             elif flips == 3:
#                 p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
#         else:
#             x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
#             if flips == 2:
#                 y = img_size[0] - y  # de-flip ud
#             elif flips == 3:
#                 x = img_size[1] - x  # de-flip lr
#             p = torch.cat((x, y, wh, p[..., 4:]), -1)
#         return p
#
#     def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
#         for m in self.model.modules():
#             if type(m) is Conv and hasattr(m, 'bn'):
#                 m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
#                 delattr(m, 'bn')  # remove batchnorm
#                 m.forward = m.fuseforward  # update forward
#         self.info()
#         return self


# class ModelA(nn.Module):
#     '''
#     这个和model的结构一样，但是不方便加载训练好的模型参数

#     在v5中加入attention
#     只能够使用默认参数， 不支持传入nc和anchors
#     '''
#     def __init__(self, cfg='yolov5_official/yamls/yolov5s.yaml', ch=3, nc=None, anchors=None):
#         super(ModelA, self).__init__()
#         inch = ch
#         self.yaml_file = Path(cfg).name
#         # 从yaml中模型结构参数
#         with open(cfg) as f:
#             self.yaml = yaml.safe_load(f)
#         self.anchors = self.yaml.get('anchors')
#         self.nc = self.yaml.get('nc')
#         self.gd = self.yaml['depth_multiple']
#         self.gw = self.yaml['width_multiple']

#         num_anchors = len(self.anchors[0]) // 2
#         outch = num_anchors * (self.nc + 5)

#         self.backbone1 = nn.Sequential(
#             Focus(3, 32),
#             Conv(32, 64, 3, 2),  # stride2
#             C3(64, 64, n=1),
#             Conv(64, 128, 3, 2),
#             C3(128, 128, n=3),
#         )

#         self.backbone2 = nn.Sequential(
#             Conv(128, 256, 3, 2),
#             C3(256, 256, n=3),
#         )

#         self.neck1 = nn.Sequential(
#             Conv(256, 512, 3, 2),
#             SPP(512, 512),
#             C3(512, 512, shortcut=False, n=1),
#             Conv(512, 256, 1, 1),
#         )

#         self.upsample = nn.Upsample(None, 2, 'nearest')
#         self.upsample2 = nn.Upsample(None, 2, 'nearest')

#         self.neck2 = nn.Sequential(
#             C3(512, 256, n=1, shortcut=False),
#             Conv(256, 128, 1, 1)
#         )

#         self.head_top1 = C3(256, 128, n=1, shortcut=False)
#         self.connect_conv1 = Conv(128, 128, 3, 2)
#         self.head_mid1 = C3(256, 256, n=1, shortcut=False)
#         self.connect_conv2 = Conv(256, 256, 3, 2)
#         self.head_bot1 = C3(512, 512, n=1, shortcut=False)


#     def forward(self, x, augment=False):
#         '''
#         x 是输入样本 augment不能赋值，只是为了保持和model一致
#         '''
#         y1 = self.backbone1(x)
#         y2 = self.backbone2(y1)
#         y3 = self.neck1(y2)
#         y4 = self.upsample(y3)
#         y4 = torch.cat([y4, y2], dim=1)
#         y4 = self.neck2(y4)
#         y5 = self.upsample2(y4)
#         y5 = torch.cat([y5, y1], dim=1)
#         y5 = self.head_top1(y5)  # 送给 76*76的conv 并且会给到mid 分支

#         y5e = self.connect_conv1(y5)
#         y5e = torch.cat([y5e, y4], dim=1)
#         y6 = self.head_mid1(y5e)

#         y6e = self.connect_conv2(y6)
#         y6e = torch.cat([y6e, y3], dim=1)
#         y7 = self.head_bot1(y6e)

#         print(y5.shape, y6.shape, y7.shape)
#         return [y5, y6, y7]


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


# 从 yaml 文件中解析 model
def parse_model(par, inch):
    # depth_multiple 和 width_multiple 用于对layer中的 channels 进行缩放
    anchors, nc, gd, gw = par['anchors'], par['nc'], par['depth_multiple'], par['width_multiple']
    num_anchors = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors  # 3 有几个anchors
    out_channels = num_anchors * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, c2, save = [], inch[-1], []
    # [from, number, module, args]  读取模型，有23层
    for i, (f, n, m, args) in enumerate(par['backbone'] + par['head']):
        # print(i, (f, n, m, args))  # 形式已经在model的init中列出， 其中m是模块名称， arg是模块参数
        m = eval(m) if isinstance(m, str) else m  # 将str转成 <class 'yolov5_official.modules.Concat'> class 类型
        # 尽可能将所有的module参数转为float
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # 通道缩放
        if m in [Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3, C3TR]:
            # print('inch:', inch)
            c1, c2 = inch[f], args[0]  # module的输入输出维度

            if c2 != out_channels:  # 如果当前输出不是最终输出维度
                c2 = make_divisible(c2 * gw, 8)  # 对维度进行缩放

            args = [c1, c2, *args[1:]]  # 输入输出维度以及后面的维度
            if m in [BottleneckCSP, C3, C3TR]:  # 如果是这三种module
                args.insert(2, n)  # 需要插入n 来决定 内部的layer repeat多少次
                n = 1
        elif m is nn.BatchNorm2d:
            args = [inch[f]]
        elif m is Concat:
            c2 = sum([inch[x] for x in f])
        elif m is Detect:
            args.append([inch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:  # 维度缩小后的c
            c2 = inch[f] * args[0] ** 2
        elif m is Expand:  # 维度扩张后的c
            c2 = inch[f] // args[0] ** 2
        else:
            c2 = inch[f]

        # print(m, args)  #<class 'yolov5_official.modules.Conv'> [256, 256, 3, 2]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # 是否要重复多次
        t = str(m)[8:-2].replace('__main__.', '')
        npp = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, npp
        # f可以是-1这种int 也可能是一个list  选择适当x，在之后进行保存
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 当x不为-1， f不为-1的int或者list中不为-1的项
        # print('save: ', save)  # save:  [6, 4, 14, 10]
        layers.append(m_)
        if i == 0:  # 如果是网络的第一层，我们重置inch为空list
            inch = []  # 用于存储每个layer的输出
        inch.append(c2)

    return nn.Sequential(*layers), sorted(save)


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    # 向上取整再乘上divisor
    return math.ceil(x / divisor) * divisor


# 权重初始化
def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:  # 卷积权重不初始化
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.upsample(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


if __name__ == '__main__':
    import yolov5_official.config as cfg

    # 读取超参数和数据参数
    with open(cfg.hyp) as f:
        hyperparams = yaml.safe_load(f)
    with open(cfg.data) as f:
        dataparams = yaml.safe_load(f)

    model = Model(cfg.cfg, ch=3, nc=dataparams.get('nc'), anchors=hyperparams.get('anchors', None))

'''
{'epoch': -1,

 'best_fitness': array([0.37287207]),

 'training_results': None,

 'model': Model(

   (model): Sequential(

     (0): Focus(

       (conv): Conv(

         (conv): Conv2d(12, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

         (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

     )

     (1): Conv(

       (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

       (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

       (act): SiLU()

     )

     (2): C3(

       (cv1): Conv(

         (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv2): Conv(

         (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv3): Conv(

         (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (m): Sequential(

         (0): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

       )

     )

     (3): Conv(

       (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

       (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

       (act): SiLU()

     )

     (4): C3(

       (cv1): Conv(

         (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv2): Conv(

         (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv3): Conv(

         (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (m): Sequential(

         (0): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

         (1): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

         (2): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

       )

     )

     (5): Conv(

       (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

       (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

       (act): SiLU()

     )

     (6): C3(

       (cv1): Conv(

         (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv2): Conv(

         (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv3): Conv(

         (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (m): Sequential(

         (0): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

         (1): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

         (2): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

       )

     )

     (7): Conv(

       (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

       (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

       (act): SiLU()

     )

     (8): SPP(

       (cv1): Conv(

         (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv2): Conv(

         (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (m): ModuleList(

         (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)

         (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)

         (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)

       )

     )

     (9): C3(

       (cv1): Conv(

         (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv2): Conv(

         (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv3): Conv(

         (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (m): Sequential(

         (0): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

       )

     )

     (10): Conv(

       (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

       (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

       (act): SiLU()

     )

     (11): Upsample(scale_factor=2.0, mode=nearest)

     (12): Concat()

     (13): C3(

       (cv1): Conv(

         (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv2): Conv(

         (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv3): Conv(

         (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (m): Sequential(

         (0): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

       )

     )

     (14): Conv(

       (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

       (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

       (act): SiLU()

     )

     (15): Upsample(scale_factor=2.0, mode=nearest)

     (16): Concat()

     (17): C3(

       (cv1): Conv(

         (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv2): Conv(

         (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv3): Conv(

         (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (m): Sequential(

         (0): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

       )

     )

     (18): Conv(

       (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

       (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

       (act): SiLU()

     )

     (19): Concat()

     (20): C3(

       (cv1): Conv(

         (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv2): Conv(

         (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv3): Conv(

         (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (m): Sequential(

         (0): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

       )

     )

     (21): Conv(

       (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

       (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

       (act): SiLU()

     )

     (22): Concat()

     (23): C3(

       (cv1): Conv(

         (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv2): Conv(

         (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (cv3): Conv(

         (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

         (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

         (act): SiLU()

       )

       (m): Sequential(

         (0): Bottleneck(

           (cv1): Conv(

             (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

             (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

           (cv2): Conv(

             (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

             (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)

             (act): SiLU()

           )

         )

       )

     )

     (24): Detect(

       (m): ModuleList(

         (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))

         (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))

         (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))

       )

     )

   )

 ),

 'optimizer': None,

 'wandb_id': None}
'''
