import os
import torch
import torch.nn as nn
from utils.lossUtils import loss
from utils.moduleUtils import SPP
from utils.utils import decode_boxes, iou_score, create_gridv3


# 自定义SiLU
class SiLU(torch.nn.Module):
    @staticmethod
    def forward(x):
        return x * nn.Sigmoid()(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1, bias=False, e=1.):
        super(ConvBlock, self).__init__()
        in_ch = round(e * in_ch)
        out_ch = round(out_ch * e)
        if p is None:
            p = k // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=bias),
            nn.BatchNorm2d(out_ch),
            SiLU()
        )

    def forward(self, x):
        return self.conv(x)


class Focus(nn.Module):
    # v5 独有操作，操作如图所示
    # 以Yolov5s的结构为例，原始608 608 3的图像输入Focus结构，采用切片操作，先变成304 304 12的特征图，再经过一次32个卷积核的卷积操作，最终变成304 304 32的特征图。
    def __init__(self, cin, cout, k=3, s=1, p=1, g=1, e=1.0):
        super(Focus, self).__init__()
        cout = round(cout * e)
        self.main = ConvBlock(cin * 4, cout, k, s, p, g=g)

    def forward(self, x):
        # 这个操作完成了 如图的focus切片操作
        flat = torch.cat([
            x[..., 0::2, 0::2],  # x [...,0::2, 0::2] = x[:, :, 0::2, 0::2], 也就是前面全都是 :
            x[..., 1::2, 0::2],  # 1::2 表示一种切片方法，即从idx=1开始切片，步长为2
            x[..., 0::2, 1::2],  # 0::2 表示从idx=0开始取值，步长为2
            x[..., 1::2, 1::2]
        ], 1)

        return self.main(flat)


class CSPResUnit(nn.Module):
    def __init__(self, cin, cout, n):
        # 相比于v4在conv上增加了bn和act
        super(CSPResUnit, self).__init__()
        self.shortcut = cin == cout
        res = nn.Sequential(
            ConvBlock(cin, cin, 1, 1, 0),
            ConvBlock(cin, cout, 3, 1, 1)
        )
        self.main = nn.Sequential(*[res for _ in range(n)])

    def forward(self, x):
        return x + self.main(x) if self.shortcut else self.main(x)


class CSPModule1(nn.Module):
    # 新的CSP module 相对于v4使用的有所改变
    def __init__(self, cin, cout, k=1, s=1, p=None, g=1, n=1, e=None):
        super(CSPModule1, self).__init__()
        if e is None:
            e = [0.5, 0.5]
        cin = round(cin * e[1])  # 这里对维度进行了改变，乘上了一个系数，初始0.5
        cout = round(cout * e[1])  # 这里对维度进行了改变，乘上了一个系数，初始0.5
        hdim = cout // 2
        n = round(n * e[0])  # 这里block数目进行了改变

        if p is None:
            p = k // 2
        self.cbl = ConvBlock(cin, hdim, k, s, p, g=1)
        self.res = CSPResUnit(hdim, hdim, n)

        # res后面的conv在最新的结构中去掉了，只有一个conv，在下支路
        self.conv = nn.Conv2d(cin, hdim, 1, 1, 0, bias=False)
        self.last = nn.Sequential(
            nn.BatchNorm2d(hdim * 2),
            nn.LeakyReLU(inplace=True),
            ConvBlock(hdim * 2, cout, 1, 1, 0)
        )

    def forward(self, x):
        y1 = self.cbl(x)
        y1 = self.res(y1)
        y2 = self.conv(x)
        out = torch.cat([y1, y2], 1)
        out = self.last(out)
        return out


class CSPModule2(nn.Module):
    def __init__(self, cin, cout, e=.5, n=1):
        super(CSPModule2, self).__init__()
        cin = round(cin * e)
        cout = round(cout * e)
        hdim = cin // 2
        self.start = ConvBlock(cin, hdim, 1, 1, 0)
        recur_block = nn.Sequential(
            ConvBlock(hdim, hdim, 1, 1, 0),
            ConvBlock(hdim, hdim, 3, 1, 1)
        )
        self.main = nn.Sequential(*[recur_block for _ in range(n)])
        self.end = nn.Conv2d(hdim, hdim, 1, 1, 0, bias=False)

        self.conv = nn.Conv2d(cin, hdim, 1, 1, 0, bias=False)

        self.last = nn.Sequential(
            nn.BatchNorm2d(hdim * 2),
            nn.LeakyReLU(inplace=True),
            ConvBlock(hdim * 2, cout, 1, 1, 0)
        )

    def forward(self, x):
        y1 = self.start(x)
        y1 = self.main(y1)
        y1 = self.end(y1)
        y2 = self.conv(x)

        y = torch.cat([y1, y2], 1)
        y = self.last(y)
        return y


class CSPDarknet(nn.Module):
    def __init__(self, e1=0.33, e2=0.5):
        super(CSPDarknet, self).__init__()
        self.main = nn.Sequential(
            Focus(3, 64, e=e2),
            ConvBlock(round(64 * e2), round(128 * e2), k=3, s=2, p=1),
            CSPModule1(128, 128, n=3, e=[e1, e2]),
            ConvBlock(round(128 * e2), round(256 * e2), k=3, s=2, p=1),
            CSPModule1(256, 256, n=9, e=[e1, e2]),
        )

    def forward(self, x):
        y = self.main(x)
        return y


class SPPModule(nn.Module):
    def __init__(self, cin, cout, e=1.):
        super(SPPModule, self).__init__()
        cin = round(e * cin)
        cout = round(e * cout)
        hdim = cin // 2
        self.cbl1 = ConvBlock(cin, hdim, 1, 1)
        self.spp = SPP()
        self.cbl2 = ConvBlock(hdim * 4, cout, 1, 1)

    def forward(self, x):
        y = self.cbl1(x)
        y = self.spp(y)
        y = self.cbl2(y)
        return y


class YOLOv5(nn.Module):
    '''
    完全根据结构图制作yolov5模型,  这个版本是v5s
    '''

    def __init__(self, nc=80, e1=0.33, e2=0.5, anchors=None, pretrained=False, input_size=3, stride=None):
        super(YOLOv5, self).__init__()
        if anchors is None:
            anchors = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
        self.num_anchors = len(anchors) // 3
        self.num_classes = nc
        self.anchor_size = torch.Tensor(anchors).view(3, len(anchors) // 3, 2)
        self.num_anchors = len(anchors) // 3  # self.anchor_size.size(1)
        self.input_size = input_size
        self.stride = stride if stride is not None else [8, 16, 32]
        self.trainable = True

        # 创建网格
        if anchors is not None:
            self.grid_cell, self.stride_tensor, self.all_anchors_wh = create_gridv3(input_size, strides=self.stride,
                                                                                    num_anchors=self.num_anchors,
                                                                                    anchor_size=self.anchor_size)

        self.base = CSPDarknet(e1, e2)

        if pretrained:
            path_to_dir = os.path.dirname(os.path.abspath(__file__))
            print('Loading the darknet')
            self.base.load_state_dict(torch.load(path_to_dir))

        # up代表上支路
        self.first_layer = nn.Sequential(
            ConvBlock(256, 512, 3, 2, 1, e=e2),
            CSPModule1(512, 512, n=9, e=[e1, e2])
        )

        self.second_layer = nn.Sequential(
            ConvBlock(512, 1024, k=3, s=2, p=1, e=e2),
            SPPModule(1024, 1024, e=e2),
            CSPModule2(1024, 1024, n=1, e=e2),
            ConvBlock(1024, 512, 1, 1, 0, e=e2)
        )

        self.upsample_medium = nn.Upsample(scale_factor=2)

        self.third_layer = nn.Sequential(
            CSPModule2(1024, 512, n=1, e=e2),
            ConvBlock(512, 256, 1, 1, 0, e=e2)
        )

        self.upsample_large = nn.Upsample(scale_factor=2)

        self.up = CSPModule2(512, 256, n=1, e=e2)
        self.up2middle = ConvBlock(256, 256, 3, 2, 1, e=e2)
        self.middle = CSPModule2(512, 512, n=1, e=e2)
        self.middle2down = ConvBlock(512, 512, 3, 2, 1, e=e2)
        self.down = CSPModule2(1024, 1024, n=1, e=e2)

        #  detection
        self.det_large = nn.Conv2d(round(e2 * 256), 3 * (5 + nc), 1, 1, 0)
        self.det_medium = nn.Conv2d(round(512 * e2), 3 * (5 + nc), 1, 1, 0)
        self.det_small = nn.Conv2d(round(1024 * e2), 3 * (5 + nc), 1, 1, 0)

    def forward(self, x):
        y1 = self.base(x)  # 从FPN中取出大中小三种特征

        y2 = self.first_layer(y1)

        y3 = self.second_layer(y2)

        y4 = self.upsample_medium(y3)
        y4 = torch.cat([y4, y2], 1)
        y4 = self.third_layer(y4)

        y5 = self.upsample_large(y4)
        y5 = torch.cat([y5, y1], 1)
        y5 = self.up(y5)

        yl = self.det_large(y5)

        y6 = self.up2middle(y5)
        y6 = torch.cat([y6, y4], 1)
        y6 = self.middle(y6)

        ym = self.det_medium(y6)

        y7 = self.middle2down(y6)
        y7 = torch.cat([y7, y3], 1)
        y7 = self.down(y7)

        ys = self.det_small(y7)

        return ys, ym, yl

    def computeLoss(self, preds, target=None, eva=False):
        total_conf_pred = []
        total_cls_pred = []
        total_txtytwth_pred = []
        b = hw = 0
        for pred in preds:
            bs, anc, h, w = pred.size()
            # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(bs, h * w, anc)
            # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
            # [B, H*W*anchor_n, 1]   conf_pred是每个anchor的是否存在objectness置信度
            conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(bs, h * w * self.num_anchors, 1)
            cls_pred = pred[:, :, 1 * self.num_anchors:(1 + self.num_classes) * self.num_anchors].contiguous().view(bs,
                                                                                                                    h * w * self.num_anchors,
                                                                                                                    self.num_classes)
            bb_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous().view(bs,
                                                                                               h * w * self.num_anchors,
                                                                                               -1)

            total_conf_pred.append(conf_pred)
            total_cls_pred.append(cls_pred)
            total_txtytwth_pred.append(bb_pred)

            b = bs
            # 累加所有的proposal boxes
            hw += h * w

        total_conf_pred = torch.cat(total_conf_pred, 1)
        total_cls_pred = torch.cat(total_cls_pred, 1)
        total_txtytwth_pred = torch.cat(total_txtytwth_pred, 1)

        if self.trainable and not eva:
            bb_pred = total_txtytwth_pred.view(b, hw, self.num_anchors, 4)

            # 从txtytwth预测中解算出x1y1x2y2坐标  xy坐标形式的bb pred和gt
            # txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]->x1y1x2y2_pred : [B, H*W, anchor_n, 4] containing [xmin, ymin, xmax, ymax]
            xyxy_pred = (decode_boxes(bb_pred, self.all_anchors_wh, self.grid_cell) / self.input_size).view(-1, 4)
            xyxy_gt = target[:, :, 7:].view(-1, 4)  # -1是B*H*W*anchor_num
            # 计算iou分数
            iou_pred = iou_score(xyxy_pred, xyxy_gt).view(b, -1, 1)
            # 避免iou_pred将梯度回传
            with torch.no_grad():
                gt_conf = iou_pred.clone()

            # 我们讲pred box与gt box之间的iou作为objectness的学习目标.
            # [obj, cls, txtytwth, scale_weight, x1y1x2y2] -> [conf, obj, cls, txtytwth, scale_weight]
            target = torch.cat([gt_conf, target[:, :, :7]], 2)
            xyxy_pred = xyxy_pred.view(b, -1, 4)

            conf_loss, cls_loss, bbox_loss, iou_loss = loss(pred_conf=total_conf_pred,
                                                            pred_cls=total_cls_pred,
                                                            pred_txtytwth=xyxy_pred,
                                                            pred_iou=iou_pred,
                                                            label=target)
            return conf_loss, cls_loss, bbox_loss, iou_loss
        else:
            return None


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
