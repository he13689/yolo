'''
TinyYOLOv3只用了两个尺度，stride=32和stride=16，这里需要注意一点的是512层和1024层之间的maxpooling的stride=1，
而不是2，因此，为了保证输入输出不改变feature map的宽高，需要补左0右1、上0下1的zero padding
'''
import os

import torch
import torch.nn as nn
from models.basemodel.d19 import DarkNet_Light
import numpy as np
from config import *
from utils.lossUtils import loss
from utils.moduleUtils import Conv, reorg_layer
from utils.utils import decode_boxes, iou_score, postprocess, create_gridv3

from models.basemodel.d19 import DarkNet_53


class YOLOv3(nn.Module):
    def __init__(self, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.50,
                 anchor_size=None, pretrained=False, hr=False):
        super(YOLOv3, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = [16, 32]  # 多stride，分成
        self.anchor_size = torch.Tensor(anchor_size).view(2, len(anchor_size) // 2, 2)
        self.num_anchors = self.anchor_size.size(1) # 2

        # 创建网格
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = create_gridv3(input_size, strides=self.stride,
                                                                                num_anchors=self.num_anchors,
                                                                                anchor_size=self.anchor_size)

        # backbone darknet-53
        self.base = DarkNet_Light()  # 输出维度1024
        if pretrained:
            print('Loading the pretrained model ...')
            path_to_dir = os.path.dirname(os.path.abspath(__file__))
            if hr:
                self.base.load_state_dict(
                    torch.load(path_to_dir + '/weights/darknet_tiny_hr_61.85.pth'), strict=False)
            else:
                self.base.load_state_dict(
                    torch.load(path_to_dir + '/weights/darknet_tiny_63.50_85.06.pth'), strict=False)

        # s = 32
        self.conv_set_2 = Conv(1024, 256, k=3, p=1)
        self.conv_1x1_2 = Conv(256, 128, k=1)
        self.extra_conv_2 = Conv(256, 512, k=3, p=1)
        self.pred_2 = nn.Conv2d(512, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # s = 8
        self.conv_set_1 = Conv(384, 256, k=3, p=1)
        self.pred_1 = nn.Conv2d(256, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

    def set_grid(self, x):
        self.input_size = x
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = create_gridv3(x, self.stride, self.num_anchors,
                                                                                self.anchor_size)

    def forward(self, x, target=None, eva=False):
        C_4, C_5 = self.base(x)
        y1 = self.conv_set_2(C_5)
        y1out = self.upsample2(self.conv_1x1_2(y1))

        y2 = torch.cat([C_4, y1out], 1)
        y2 = self.conv_set_1(y2)
        y2out = self.pred_1(y2)
        
        y1out = self.extra_conv_2(y1)
        y1out = self.pred_2(y1out)

        y = [y2out, y1out]

        total_conf_pred = []
        total_cls_pred = []
        total_txtytwth_pred = []
        b = hw = 0  #

        for pred in y:  # 依次从小的感受野（感受区域较小，但是数量更多）到大
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
            xyxy_pred = (decode_boxes(bb_pred, self.all_anchors_wh, self.grid_cell, v2=False) / self.input_size).view(-1, 4)
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
            xyxy_pred = total_txtytwth_pred.view(b, hw, self.num_anchors, 4)
            with torch.no_grad():
                # batch size = 1
                # 测试时，笔者默认batch是1，
                # 因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W*num_anchor, 1] -> [H*W*num_anchor, 1]
                conf_pred = nn.Sigmoid()(total_conf_pred)[0]
                bboxes = torch.clamp(
                    (decode_boxes(xyxy_pred, self.all_anchors_wh, self.grid_cell, v2=False) / self.input_size)[0], 0., 1.)
                scores = self.softmax(total_cls_pred[0, :, :]) * conf_pred

                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                bboxes, scores, cls_inds = postprocess(bboxes, scores, self.conf_thresh, self.num_classes,
                                                       self.nms_thresh)
                return bboxes, scores, cls_inds
