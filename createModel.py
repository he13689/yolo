import torch
import yaml
import torch.nn as nn

from yolov5_official.model import Detect
from yolov5_official.utils import *
import math
from yolov5_official.modules import *


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

        # 一块一块的建立模型，每次加入一个m
        # print(m, args)  #<class 'yolov5_official.modules.Conv'> [256, 256, 3, 2]
        m = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # 是否要重复多次
        m.f, m.i, m.np = f, i, sum([x.numel() for x in m.parameters()])
        # f可以是-1这种int 也可能是一个list  选择适当x，在之后进行保存
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 当x不为-1， f不为-1的int或者list中不为-1的项
        # print('save: ', save)  # save:  [6, 4, 14, 10]
        layers.append(m)
        if i == 0:  # 如果是网络的第一层，我们重置inch为空list
            inch = []  # 用于存储每个layer的输出， 作为下个layer的输入
        inch.append(c2)
        # print('mf: ', m.f)  # mf:  [-1, 10] or mf:  -1

    return nn.Sequential(*layers), sorted(save)


def load_model():
    with open('yolov5_official/yamls/yolov5s.yaml') as f:
        yaml_model = yaml.safe_load(f)



if __name__ == '__main__':
    print('loading model')
