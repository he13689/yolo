import os
import torch
import torch.nn as nn

from utils.lossUtils import loss
from utils.moduleUtils import Conv, reorg_layer
from models.basemodel.d19 import Dark19
import numpy as np
from utils.utils import nms, postprocess, create_grid
from models.model_config import *
from utils.utils import decode_boxes, iou_score


# 使用darknet作为backbone的第二版yolo
class YOLOv2(nn.Module):
    def __init__(self, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.5,
                 anchor_size=None, hr=False, pretrain=False):
        super(YOLOv2, self).__init__()
        self.input_size = input_size  # 输入大小
        self.num_classes = num_classes  # 分类 voc 20
        self.trainable = trainable
        self.conf_thresh = conf_thresh  # 置信度门槛
        self.nms_thresh = nms_thresh  # nms门槛
        self.anchor_size = torch.Tensor(anchor_size) if input_size == 640 else torch.Tensor(anchor_size)/2
        self.num_anchors = len(anchor_size)  # 锚点数量
        self.stride = 32  # if input_size==640 else 16 不能这么使用，因为会导致输出的pred大小和create的grid的大小不一致 划分grid的步长 网格中每个小网格的长和宽

        self.grid_cell, self.all_anchor_wh = create_grid(input_size, self.stride, self.anchor_size)  # 建立网格 大小是input_size
        self.base = Dark19()
        
        # 是否加载预计模型
        if pretrain:
            print(f'loading pretrained d19 net {"with hr" if hr else "without hr"} ......')
            path_to_dir = os.path.dirname(os.path.abspath(__file__))  # 去掉文件名，返回目录  __file__表示了当前文件的path
            if hr: # 是否使用高分辨率图像
                self.base.load_state_dict(torch.load(path_to_dir + '/weights/darknet19_hr_75.52_92.73.pth'),
                                          strict=False)
            else:
                self.base.load_state_dict(torch.load(path_to_dir + '/weights/darknet19_72.96.pth'), strict=False)

        # detection head 探测头
        self.convsets1 = nn.Sequential(
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        self.route_layer = Conv(512, 64, k=1)
        # reorg_layer 将 原来 80X80的图像变成40*40，同时将h w维度缩小的维度变到channel维度
        self.reorg = reorg_layer(stride=2)
        
        # 1280 = 64*4 + 1024
        self.convsets2 = Conv(1280, 1024, k=3, p=1)
        # conf xywh classes
        self.pred = nn.Conv2d(1024, self.num_anchors * (1 + 4 + self.num_classes), 1)

    def set_grid(self, size):
        self.input_size = size
        self.grid_cell, self.all_anchor_wh = create_grid(size, self.stride, self.anchor_size)  # 创建 grid

    def forward(self, x, target=None, eva=False):
        _, c5, c6 = self.base(x)  # 通过训练好的dark net进行
        # head
        p6 = self.convsets1(c6)
        # 处理c5特征, c5特征比c6要大， 因此先用
        p5 = self.reorg(self.route_layer(c5))
        # 融合两种特征
        p6 = torch.cat([p5, p6], dim=1)
        # head 将两种融合后的特征进行处理
        p6 = self.convsets2(p6)

        # 预测
        pred = self.pred(p6)
#        print(pred.size())  # torch.Size([1, 125, 10, 10])  torch.Size([1, 125, 20, 20])

        bs, c, h, w = pred.size()

        pred = pred.permute(0, 2, 3, 1).contiguous().view(bs, h * w, c)
        conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(bs, h * w * self.num_anchors, 1)
        cls_pred = pred[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(bs,
                                                                                                                 h * w * self.num_anchors,
                                                                                                                 self.num_classes)
        xywh_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
        

        if self.trainable and not eva:  # train 如果是在训练阶段，那么直接会把损失返回
            xywh_pred = xywh_pred.view(bs, h * w, self.num_anchors, 4)
            # decode box
            xxyy_pred = (decode_boxes(xywh_pred, self.all_anchor_wh, self.grid_cell) / self.input_size).view(-1, 4)
            xxyy_gt = target[:, :, 7:].view(-1, 4)
            # 计算预测框和真实框之间的IoU
            iou_pred = iou_score(xxyy_pred, xxyy_gt).view(bs, -1, 1)
            # 不计算梯度
            with torch.no_grad():
                # 将IoU作为置信度的学习目标
                gt_conf = iou_pred.clone()

            xywh_pred = xywh_pred.view(bs, h * w * self.num_anchors, 4)
            target = torch.cat([gt_conf, target[:, :, :7]], dim=2)
            # 计算损失
            conf_loss, cls_loss, bbox_loss, iou_loss = loss(conf_pred, cls_pred, xywh_pred, iou_pred, label=target)
            return conf_loss, cls_loss, bbox_loss, iou_loss
        else:  # test
            # batch size = 1
            # 测试时，笔者默认batch是1，
            # 因此，我们不需要用batch这个维度，用[0]将其取走。
            # [B, H*W*num_anchor, 1] -> [H*W*num_anchor, 1]
            xywh_pred = xywh_pred.view(bs, h * w, self.num_anchors, 4)
            with torch.no_grad():
                conf_pred = nn.Sigmoid()(conf_pred)[0]
                bb = torch.clamp((decode_boxes(xywh_pred, all_anchor_wh=self.all_anchor_wh, grid_cell=self.grid_cell) / self.input_size)[0], 0., 1.)
                scores = nn.Softmax(dim=1)(cls_pred[0, :, :]) * conf_pred

                # 将预测放在cpu处理上，以便进行后处理
                scores = scores.cpu().numpy()
                bb = bb.cpu().numpy()
                # 后期处理
                bb, scores, cls_index = postprocess(bb, scores, self.conf_thresh, num_classes=self.num_classes,
                                                    nms_threshold=self.nms_thresh)
                # 返回bb、分数和索引
                return bb, scores, cls_index
