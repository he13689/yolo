import torch
import torch.nn as nn
from models.basemodel import *
import numpy as np
from config import *
from utils.lossUtils import loss
from utils.moduleUtils import Conv, reorg_layer
from utils.utils import create_grid, decode_boxes, iou_score, postprocess
from models.basemodel.d19 import DarkNet_Tiny

# yolo 的 slim 版本，更加纤细， 网络内部参数量更少，基础网络是DarkNet_Tiny
class YOLOv2(nn.Module):
    def __init__(self, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, hr=False):
        super(YOLOv2, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.Tensor(anchor_size) if input_size == 640 else torch.Tensor(anchor_size)/2
        self.num_anchors = len(anchor_size)
        self.stride = 32 # if input_size == 640 else 16
        # init set
        self.grid_cell, self.all_anchor_wh = create_grid(input_size, self.stride, self.anchor_size)

        # detection head 512dim
        self.convset1 = nn.Sequential(
            Conv(512, 512, k=3, p=1),
            Conv(512, 512, k=3, p=1)
        )

        self.route_layer = Conv(256, 32, k=1)
        # 重新整理
        self.reorg = reorg_layer(stride=2)
        # 512+32*2=640
        self.convset2 = Conv(640, 512, k=3, p=1)
        self.base = DarkNet_Tiny()

        # prediction layer
        self.pred = nn.Conv2d(512, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = create_grid(input_size, self.stride, self.anchor_size)

    def forward(self, x, target=None, eva=False):
        _, c5, c6 = self.base(x)  # 通过训练好的dark net进行
        # head
        p6 = self.convset1(c6)
        # 处理c5特征
        p5 = self.reorg(self.route_layer(c5))
        # 融合两种特征
        p6 = torch.cat([p5, p6], dim=1)
        # head 将两种融合后的特征进行处理
        p6 = self.convset2(p6)

        # 预测
        pred = self.pred(p6)

        bs, c, h, w = pred.size()

        pred = pred.permute(0, 2, 3, 1).contiguous().view(bs, h * w, c)
        conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(bs, h * w * self.num_anchors, 1)
        cls_pred = pred[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(bs,
                                                                                                                 h * w * self.num_anchors,
                                                                                                                 self.num_classes)
        xywh_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()

        if self.trainable and not eva:  # train
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
                bb = torch.clamp((decode_boxes(xywh_pred, self.all_anchor_wh, self.grid_cell) / self.input_size)[0], 0., 1.)
                scores = nn.Softmax(dim=1)(cls_pred[0, :, :]) * conf_pred

                # 将预测放在cpu处理上，以便进行后处理
                scores = scores.cpu().numpy()
                bb = bb.cpu().numpy()
                # 后期处理
                bb, scores, cls_index = postprocess(bb, scores, self.conf_thresh, num_classes=self.num_classes,
                                                    nms_threshold=self.nms_thresh)
                # 返回bb、分数和索引
                return bb, scores, cls_index



