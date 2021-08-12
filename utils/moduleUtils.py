import math
import torch
import torch.nn as nn
from copy import deepcopy


# 对卷积块进行封装，可选是否包括activation func
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, g=1, act=True):
        super(Conv, self).__init__()
        if act:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=g),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=g),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        return self.conv(x)


# reshape 并且 根据给定 stride 改变维度顺序
class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        bs, c, h, w = x.size()
        hgrid, wgrid = h // self.stride, w // self.stride

        x = x.view(bs, c, hgrid, self.stride, wgrid, self.stride).transpose(3, 4).contiguous()
        x = x.view(bs, c, hgrid * wgrid, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(bs, c, self.stride * self.stride, wgrid * hgrid).transpose(1, 2).contiguous()
        x = x.view(bs, -1, hgrid, wgrid)
        return x


# 残差块
class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super(resblock, self).__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            res_block = nn.Sequential(
                Conv(ch, ch // 2, 1),
                Conv(ch // 2, ch, 3, 1)
            )
            self.module_list.append(res_block)

    def forward(self, x):
        for mod in self.module_list:
            x = mod(x) + x
        return x


# 进行三次maxpool， 每次pool的视野都不一样，将输入和pool后的结果cat在一起
class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.mp1 = nn.MaxPool2d(5, 1, 2)
        self.mp2 = nn.MaxPool2d(9, 1, 4)
        self.mp3 = nn.MaxPool2d(13, 1, 6)

    def forward(self, x):
        x_1 = self.mp1(x)
        x_2 = self.mp2(x)
        x_3 = self.mp3(x)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)
        return x


# CSP 残差块的in out channels是不变的
class CSPResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super(CSPResBlock, self).__init__()
        if out_ch is None:
            out_ch = in_ch
        self.conv1 = Conv(in_ch, in_ch, k=1)
        self.conv2 = Conv(in_ch, out_ch, k=3, p=1, act=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        h = self.conv2(self.conv1(x))
        out = self.act(x + h)
        return out


# 两个conv+一个residual 对应Darknet53结构图中的黑色框框
class CSPStage(nn.Module):
    def __init__(self, c, n=1):
        super(CSPStage, self).__init__()
        h_dim = c // 2
        self.conv1 = Conv(c, h_dim, 1)  # 1 x 1 卷积 降维
        self.conv2 = Conv(c, h_dim, 1)  # 1 x 1 卷积 降维
        self.resblock = nn.Sequential(*[CSPResBlock(in_ch=h_dim) for _ in range(n)])
        self.conv3 = Conv(2 * h_dim, c, 1)  # 1 x 1 卷积 通道不降维

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.resblock(self.conv2(x))
        return self.conv3(torch.cat([y1, y2], dim=1))


class Bottleneck(nn.Module):
    '''
    c1：bottleneck 结构的输入通道维度；
    c2：bottleneck 结构的输出通道维度；
    shortcut：是否给bottleneck 结构添加shortcut连接，添加后即为ResNet模块；
    g：groups，通道channels分组的参数，输入通道数、输出通道数必须同时满足被groups整除；这样会先对通道进行分组之后再卷积
    e：expansion: bottleneck 结构中的瓶颈部分的通道膨胀率，使用0.5即为变为输入的1/2；

    这里的瓶颈层，瓶颈主要体现在通道数channel上面！一般1x1卷积具有很强的灵活性，这里用于降低通道数，如上面的膨胀率为0.5，若输入通道为640，那么经过1x1的卷积层之后变为320；经过3x3之后变为输出的通道数，这样参数量会大量减少！
    这里的shortcut即为图中的红色虚线，在实际中，shortcut(捷径)不一定是上面都不操作，也有可能有卷积处理，但此时，另一支一般是多个ResNet模块串联而成！这里使用的shortcut也成为identity分支，可以理解为恒等映射，另一个分支被称为残差分支(Residual分支)。
    我们常使用的残差分支实际上是1x1+3x3+1x1的结构！
    '''

    def __init__(self, cin, cout, shortcut=True, g=1, e=0.5, n=1):
        super(Bottleneck, self).__init__()
        hdim = int(cout * e)  # hidden channels
        self.conv1 = Conv(cin, hdim, k=1)
        self.conv2 = Conv(hdim, cout, k=3, p=1, g=g)  # groups = g, 将hidden通道分组后卷积
        self.shortcut = shortcut and cin == cout  # 当输入输出维度相同，加入shortcut

    def forward(self, x):
        if self.shortcut:
            y = x + self.conv2(self.conv1(x))
        else:
            y = self.conv2(self.conv1(x))
        return y


class CSPBottleneck(nn.Module):
    '''
    c1：BottleneckCSP 结构的输入通道维度；
    c2：BottleneckCSP 结构的输出通道维度；
    n：bottleneck 结构 结构的个数；
    shortcut：是否给bottleneck 结构添加shortcut连接，添加后即为ResNet模块；
    g：groups，通道分组的参数，输入通道数、输出通道数必须同时满足被groups整除；
    e：expansion: bottleneck 结构中的瓶颈部分的通道膨胀率，使用0.5即为变为输入的12；
    torch.cat((y1, y2), dim=1)：这里是指定在第1个维度上进行合并，即在channel维度上合并;
    c_：BottleneckCSP 结构的中间层的通道数，由膨胀率e决定。

    '''

    def __init__(self, cin, cout, n=1, shortcut=True, g=1, e=.5):
        super(CSPBottleneck, self).__init__()
        hdim = int(cout * e)  # 输出维度进行膨胀
        self.conv1 = Conv(cin, hdim, k=1)  # 1x1 conv
        self.conv2 = nn.Conv2d(cin, hdim, kernel_size=1, bias=False)  # 不经过bottleblock的路径
        self.conv3 = nn.Conv2d(hdim, hdim, kernel_size=1, bias=False)  # bottleblock后的conv2d
        self.conv4 = Conv(hdim * 2, cout, k=1)
        self.bottleblock = nn.Sequential(*[Bottleneck(hdim, hdim, shortcut=shortcut, g=g, e=e) for _ in range(n)])
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.bn = nn.BatchNorm2d(hdim * 2)

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.bottleblock(y1)
        y1 = self.conv3(y1)

        y2 = self.conv2(x)
        y = torch.cat([y1, y2], 1)
        y = self.act(self.bn(y))
        y = self.conv4(y)
        return y


class CSPResUnit(nn.Module):
    def __init__(self, cin, cout, n):
        # 相比于v4在conv上增加了bn和act
        super(CSPResUnit, self).__init__()
        self.shortcut = cin == cout
        res = nn.Sequential(
            nn.Conv2d(cin, cin, 1, 1, 0),
            nn.BatchNorm2d(cin),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(cin, cout, 3, 1, 1),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(inplace=True)
        )
        self.main = nn.Sequential(*[res for _ in range(n)])

    def forward(self, x):
        return x + self.main(x) if self.shortcut else self.main(x)


class CSPModule(nn.Module):
    # 新的CSP module 相对于v4使用的有所改变
    def __init__(self, cin, cout, k=1, s=1, p=None, g=1, act=True, n=1, e=None):
        super(CSPModule, self).__init__()
        if e is None:
            e = [0.5, 0.5]
        cin = round(cin * e[1])  # 这里对维度进行了改变，乘上了一个系数，初始0.5
        cout = round(cout * e[1])  # 这里对维度进行了改变，乘上了一个系数，初始0.5
        hdim = cout // 2
        n = round(n * e[0])  # 这里block数目进行了改变
        if p is None:
            p = k//2
        if act:
            self.up = nn.Sequential(
                nn.Conv2d(cin, hdim, kernel_size=k, stride=s, groups=g, padding=p, bias=False),
                nn.BatchNorm2d(hdim),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(cin, hdim, kernel_size=k, stride=s, groups=g, padding=p, bias=False),
                nn.BatchNorm2d(hdim)
            )
        self.res = CSPResUnit(hdim, hdim, n)
        self.bottom = nn.Conv2d(cin, hdim, 1, 1, 0, bias=False)
        self.tie = nn.Sequential(
            nn.BatchNorm2d(hdim * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hdim * 2, cout, 1, 1, 0, bias=False)

        )

    def forward(self, x):
        y = torch.cat([self.up(x), self.bottom(x)], 1)
        out = self.tie(y)
        return out


class Focus(nn.Module):
    # v5 独有操作，操作如图所示
    # 以Yolov5s的结构为例，原始608 608 3的图像输入Focus结构，采用切片操作，先变成304 304 12的特征图，再经过一次32个卷积核的卷积操作，最终变成304 304 32的特征图。
    def __init__(self, cin, cout, k=3, s=1, p=1, g=1, act=True, e=1.0):
        super(Focus, self).__init__()
        cout = round(cout * e)
        self.conv = nn.Conv2d(cin*4, cout, kernel_size=k, stride=s, groups=g, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(inplace=True) if act else None

    def forward(self, x):
        # 这个操作完成了 如图的focus切片操作
        flat = torch.cat([
            x[..., 0::2, 0::2],  # x [...,0::2, 0::2] = x[:, :, 0::2, 0::2], 也就是前面全都是 :
            x[..., 1::2, 0::2],  # 1::2 表示一种切片方法，即从idx=1开始切片，步长为2
            x[..., 0::2, 1::2],  # 0::2 表示从idx=0开始取值，步长为2
            x[..., 1::2, 1::2]
        ], 1)

        y = self.conv(flat)
        y = self.bn(y)
        if self.act is not None:
            y = self.act(y)

        return y
