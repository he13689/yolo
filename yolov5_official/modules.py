# YOLOv5 common modules

import math
from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn

from yolov5_official.utils import truncated_normal_, _init_vit_weights


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # 如果是1.7以上版本则直接使用silu
        # self.act = nn.LeakyReLU() if act is True else None
        self.act = nn.SiLU(inplace=True) if act is True else None

    def forward(self, x):
        if self.act is not None:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        # 线性变换矩阵
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        #  编码维度 c 以及 头num_heads
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)  # 大小不变
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class AttentionLayer(nn.Module):
    '''  这是我们网络的输入
    torch.Size([1, 128, 32, 32])
    torch.Size([1, 256, 16, 16])
    torch.Size([1, 512, 8, 8])

    我们可以将每个特征图的每个位置看做一个 embedding 维度是128   计算它的qkv
    因为是multi head 所以我们要做多个qkv  但是为了简便我们只做一个卷积，输出维度是 mh*q的维度
    我们将输出 分割为mh份，每一份的q*k然后再乘上v

    '''

    def __init__(self, cin, cout, mh=8):
        '''
            输入输出通道数相等 并且  attention的头数量16 必须是输入通道数的整数倍
        '''
        super(AttentionLayer, self).__init__()
        assert cin == cout
        # 一般的话 SA的输入输出不变
        self.q = nn.Conv2d(cin, cout, 1, 1, 0)
        self.k = nn.Conv2d(cin, cout, 1, 1, 0)
        self.v = nn.Conv2d(cin, cout, 1, 1, 0)
        self.mh = mh
        self.dim = cout // mh

        self.scale = (self.dim) ** -0.5  # 对应的是 根号dk（k的维度） 分之一
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.init_weight(self.q)
        self.init_weight(self.k)
        self.init_weight(self.v)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        q = self.q(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B * (H * W) * C
        k = self.k(x).view(batch_size, -1, height * width)  # B * C * (H * W)
        v = self.v(x).view(batch_size, channels, -1)  # B * C * (H * W)

        self_attention_map = []
        for i in range(self.mh):
            q_slice = q[:, :, i * self.dim:(i + 1) * self.dim]
            k_slice = k[:, i * self.dim:(i + 1) * self.dim]
            v_slice = v[:, i * self.dim:(i + 1) * self.dim]

            attention = torch.bmm(q_slice, k_slice)  # B * (H * W) * (H * W)
            attention = self.softmax_(attention)

            self_attention_map.append(
                torch.bmm(v_slice, attention).view(batch_size, self.dim, height, width))  # B * C/mh * H * W

        self_attention_map = torch.cat([self_attention_map], dim=1)

        return self.gamma * self_attention_map + x

    def init_weight(self, conv):
        nn.init.kaiming_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()


# 自己写attention 模块
class AttentionBlock(nn.Module):
    def __init__(self, cin, cout, nh=16, n=1):
        super(AttentionBlock, self).__init__()
        self.att = nn.Sequential(*[AttentionLayer(cin, cout, nh) for _ in range(n)])

    def forward(self, x):
        x = self.att(x)
        return x


class SA(nn.Module):
    r"""
        Create global dependence.
        Source paper: https://arxiv.org/abs/1805.08318
    """

    def __init__(self, in_channles):
        super(SA, self).__init__()
        self.in_channels = in_channles

        self.f = nn.Conv2d(in_channels=in_channles, out_channels=in_channles // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channles, out_channels=in_channles // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channles, out_channels=in_channles, kernel_size=1)
        self.softmax_ = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.init_weight(self.f)
        self.init_weight(self.g)
        self.init_weight(self.h)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        assert channels == self.in_channels

        f = self.f(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B * (H * W) * C//8
        g = self.g(x).view(batch_size, -1, height * width)  # B * C//8 * (H * W)

        attention = torch.bmm(f, g)  # B * (H * W) * (H * W)
        attention = self.softmax_(attention)

        h = self.h(x).view(batch_size, channels, -1)  # B * C * (H * W)

        self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width)  # B * C * H * W

        return self.gamma * self_attention_map + x

    def init_weight(self, conv):
        nn.init.kaiming_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()


class Bottleneck(nn.Module):  # 如果bottle neck 去掉shortcut 那么就是Conv*2 就等于CSP2中的Conv*2层
    # Standard bottleneck    这个用在CSP中作为res unit
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # 两种CSP结构
    # CSP1_X结构应用于Backbone主干网络，另一种CSP2_X结构则应用于Neck中。
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):  # 这个就是用在base net中的CSP
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # 待transformer的c3 直接继承自C3结构，在此之上加入了TransformerBlock 改变了原来的m
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super(C3SPP, self).__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # 切片 这样做增加了正样本，加快了训练和检测速度，和mAP无关
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        # 将x(b,c,w,h) -> y(b,4c,w/2,h/2)，方法是使用间隔取样，将特征图大小除以2，加到通道上
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Contract(nn.Module):  # 将wh维度缩小，同时保证参数量不变，即增加channel的维度
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):  # 和contract相反，减少channels维度以扩展wh维度
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):  # 传入dim 进行cat
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Classifier(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        '''
        x(b,c1,20,20) to x(b,c2)
        '''
        super(Classifier, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)  # 输出是(b, c1, 1, 1)
        self.flatten = nn.Flatten(c2)
        self.conv = nn.Conv2d(c1, c2, k, s, p, g)

    def forward(self, x):
        y = torch.cat([self.ap(x) for x in (x if isinstance(x, list) else [x])], 1)
        y = self.conv(y)
        y = self.flatten(y)
        return y


class PatchEmbeddingBlock(nn.Module):
    '''
    它的作用就是 将图片按照设定patch进行划分，然后做一个flatten
    img_size, patch_size, in_c, embed_dim, norm_layer
    '''

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbeddingBlock, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 将图片横竖划分多少块
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 将图片划分为多少个块
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        # 使用conv2d将输入网格化
        self.gridding = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 对分块之后的 特征做一个norm
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        y = self.gridding(x)
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        y = y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)
        if self.norm:
            y = self.norm(y)
        return y


class MHAttention(nn.Module):
    def __init__(self, dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(MHAttention, self).__init__()
        self.num_heads = num_heads  # 有几个头
        head_dim = dim // num_heads  # 每个头是多少维度  有几个head就将qkv均分成几份
        self.scale = qk_scale or head_dim ** -0.5  # 对应的是 根号dk（k的维度） 分之一
        # 输入经过线性变换得到qkv
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 生成qkv， 一个全连接层
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # drop out rate
        self.proj = nn.Linear(dim, dim)  # 拼接后，使用（全连接层）进行映射  叫做映射线性层。
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        #  [batch_size, num_patches + 1, total_embed_dim]  其中+1是加上了class token
        bs, patches, emd = x.shape
        # [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]，其中 embed_dim_per_head = patches // self.num_heads
        qkv = self.qkv(x).view(bs, patches, 3, self.num_heads, emd // self.num_heads).permute(2, 0, 3, 1, 4)
        # 格式为 batch_size, num_heads, num_patches + 1, embed_dim_per_head
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]  最后两个维度调换位置
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attention = (q @ k.transpose(-2, -1)) * self.scale
        # 对attention做softmex
        attention = nn.Softmax(dim=-1)(attention)
        attention = self.attn_drop(attention)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attention @ v).transpose(1, 2).reshape(bs, patches, emd)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def gelu(x):  # 激活函数
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MLPBlock(nn.Module):
    def __init__(self, indim, hiddim=None, outdim=None, activate=True, droprate=0.):
        super(MLPBlock, self).__init__()
        if activate:
            self.act_layer = gelu
        else:
            self.act_layer = None
        self.indim = indim
        self.hiddim = hiddim or indim
        self.outdim = outdim or indim
        self.drop = nn.Dropout(droprate)
        self.fc1 = nn.Linear(in_features=indim, out_features=hiddim)  # 将维度膨胀4倍
        self.fc2 = nn.Linear(self.hiddim, self.outdim)  # 将维度恢复

    def forward(self, x):
        x = self.fc1(x)
        if self.act_layer:
            x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    '''
    DropPath是将深度学习模型中的多分支结构随机”删除“
    '''
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob  # 保留的几率
    # tuple加法是(x.shape[0], 1,1,...,1) 其中有x.ndim - 1个1
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 可以适用于不止2维的conv  shape是一个tuple, ndim是x自带属性
    #   随机一个tensor，加上keep_prob  rand 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 就相当于保留keep_prob的x
    random_tensor.floor_()  # binarize 向下取整, 如果是0的话就相当于从x中删除
    # 将x除以keep_prob 再乘上随机tensor
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=.5):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class EncoderBlock(nn.Module):
    # 这个block会重复N次, 不改变输入输出维度
    def __init__(self, dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 activate=True,
                 norm_layer=nn.LayerNorm):
        super(EncoderBlock, self).__init__()

        norm_layer = norm_layer or nn.LayerNorm
        self.norm1 = norm_layer(dim)
        self.attn = MHAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else None
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(dim, hidden_dim, activate=activate, droprate=drop_ratio)

    def forward(self, x):
        if self.drop_path:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


# vit transformer中的 attention 方法
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbeddingBlock, norm_layer=None,
                 act_layer=True):
        '''

        :param img_size:  图像大小
        :param patch_size: 每个图像块的大小
        :param in_c:  输入通道
        :param num_classes: 分类数量
        :param embed_dim:
        :param depth:  Encoderblock重复次数
        :param num_heads:
        :param mlp_ratio:
        :param qkv_bias:
        :param qk_scale:
        :param representation_size: MLP head中pre_logits的全连接层的节点个数，也就是输出个数 即最后一个linear的输入个数
        :param drop_ratio:
        :param attn_drop_ratio: attn 的 drop out rate
        :param drop_path_ratio:
        :param embed_layer:  emb 层
        :param norm_layer:  使用的norm
        :param act_layer: 是否使用激活函数
        '''
        super(ViT, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1  # class 的维度
        norm_layer = norm_layer or partial(nn.LayerNorm,
                                           eps=1e-6)  # partial定义norm_layer，调用norm_layer时相当于nn.LayerNorm() 并将其eps=1e-6，相当于预设部分参数

        self.emb_layer = embed_layer(img_size, patch_size, in_c, embed_dim)
        num_patches = self.emb_layer.num_patches  # 有多少块

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)  # 在加入pos emb之后的 emb 需要过一个drop

        # 使用递增的 drop rotio， linspace 从0到drop_path_ratio，其中采样depth次
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            EncoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_ratio=drop_ratio,
                         attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                         norm_layer=norm_layer, activate=act_layer)
            for i in range(depth)
        ])
        self.norm_layer = norm_layer(embed_dim)  # 通过 blocks 之后的

        # 在MLP head中加上一个logits
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())]))
        else:
            self.has_logits = False

        if num_classes > 0:
            self.head = nn.Linear(self.num_features, self.num_classes)

        # 参数初始化
        truncated_normal_(self.pos_embed, std=0.02)  # 初始化emb param
        truncated_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)  # 初始化 网络参数

    def forward(self, x):
        x = self.emb_layer(x)  # 获得flatten之后patch结果
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # 将cls_token和patch_token 连接在一起
        x = self.pos_drop(x + self.pos_embed)  # 加入pos_embed 之后 drop
        x = self.blocks(x)  # 经过transformer encoder
        x = self.norm_layer(x)  # 经过norm

        y = self.head(x[:, 0])
        return y
