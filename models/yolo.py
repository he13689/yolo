# -*- coding: utf-8 -*-
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from yolov5_official.modules import *
import yaml
import torch.nn.functional as F
import numpy as np


# 用于test阶段 输出检测结果
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        '''

        :param nc:  class 数目
        :param anchors:  anchors 的不同规格
        :param ch:  ch 有多个输入维度，表示多个不同代销输入的特征图
        :param inplace:
        '''
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        # number of detection layers 和anchors数量有关 数量为3 因为对应不同特征图大小我们使用不同大小的anchor set， 不同set对应不同的detect layer
        self.nl = len(anchors)  # 一共有三组anchor， 每组anchor对应不同大小的特征图
        self.na = len(anchors[0]) // 2  # number of anchors =3 代表每个anchor set 有 三个规格
        self.grid = [torch.zeros(1)] * self.nl  # init grid [0,0,0]
        a = torch.Tensor(anchors).float().view(self.nl, -1, 2)  # (3,6) -> (3,3,2)  每个 大小的特征图对应一个anchor
        self.register_buffer('anchors', a)
        # shape(nl,na,2)  向模块添加持久缓冲区。  这通常用于注册不应被视为模型参数的缓冲区。 缓冲区可以使用给定的名称作为属性访问。 应该就是在内存中定义一个常量，同时，模型保存和加载的时候可以写入和读出
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 输出是self.no * self.na，表示每个anchor都有 no个输出
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []
        for i in range(self.nl):  # 有几组anchor， 表示输入x是 几个不同大小的特征图
            x[i] = self.m[i](x[i])  # 将对应大小的输入传入对应的module list
            bs, _, ny, nx = x[i].shape
            # [bs, 3, outputs, ny, nx]   等于将x的第二个维度拆分为3*outputs num， 意义是对于batch中的每个elem
            # 都对应一个anchor set的结果，这个结果是 class， conf， pos 组成的
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # 训练阶段将跳过这里，直接将conv输出的结果经过变换后输出
            if not self.training:  # 如果不是训练阶段， 使用AWS 推测法
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx=nx, ny=ny)

                y = x[i].sigmoid()  # 对其进行sigmoid激活函数
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i].cuda()  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20):
        grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx))
        yv, xv = torch.from_numpy(grid_y).cuda(), torch.from_numpy(grid_x).cuda()
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
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
        self.yaml_file = Path(cfg).name
        # 从yaml中读取 model 结构
        with open(cfg) as f:
            self.yaml = yaml.safe_load(f)

        # 定义 model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            # 表示用形参中的参数代替 原始的 nc
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            # 用形参中的anchor代替原始的 anchors
            print(f"Overriding model.yaml anchors")
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
        if augment:
            return self.forward_augment(x)
        else:
            return self.forward_once(x)

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x):
        y = []  # outputs
        # 这个for的作用是，首先我们在self.save中预留设定了我们需要预留的结果
        # 当结果index满足save时，将中间结果x存在y中，当用到的时候m.f != -1，就把结果读取出来
        for m in self.model:  # 遍历模型中的所有module
            if m.f != -1:  # if not from previous layer 这就是说如果我们当前m的输入是从两个以上层的输出中得到的
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # 如果m的index是在save列表中的，此时我们就取出x放到y中

        return x


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
    num_anchors = len(anchors[0]) // 2  # 3 有几个anchors
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
        m = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # 是否要重复多次
        m.f, m.i, m.np = f, i, sum([x.numel() for x in m.parameters()])
        # f可以是-1这种int 也可能是一个list  选择适当x，在之后进行保存
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 当x不为-1， f不为-1的int或者list中不为-1的项
        # print('save: ', save)  # save:  [6, 4, 14, 10]
        layers.append(m)
        if i == 0:  # 如果是网络的第一层，我们重置inch为空list
            inch = []  # 用于存储每个layer的输出
        inch.append(c2)
        # print('mf: ', m.f)  # mf:  [-1, 10] or mf:  -1

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
