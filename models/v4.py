import os
import torch
import torch.nn as nn
import numpy as np
from config import *
from utils.lossUtils import loss
from utils.moduleUtils import Conv, SPP, CSPBottleneck
from utils.utils import decode_boxes, iou_score, postprocess, create_gridv3
from models.basemodel.d19 import CSPDarknet

'''
YOLO V4就是筛选了一些从YOLO V3发布至今，被用在各式各样检测器上，能够提高检测精度的tricks，并以YOLO V3为基础进行改进的目标检测模型。YOLO V4在保证速度的同时，大幅提高模型的检测精度(当然，这是相较于YOLO V3的）

主要变化如下：
相较于YOLO V3的DarkNet53，YOLO V4用了CSPDarkNet53
相较于YOLO V3的FPN,YOLO V4用了SPP+PAN
CutMix数据增强和马赛克（Mosaic）数据增强
DropBlock正则化

它的结构主要由三部分组成
BackBone:CSPDarknet53
Neck:SPP+PAN
HEAD:YOLO HEAD

目前做检测器MAP指标的提升，都会考虑选择一个图像特征提取能力较强的backbone，且不能太大，那样影响检测的速度。YOLO V4中，则是选择了具有CSP（Cross-stage partial connections）的darknet53,而是没有选择在imagenet上跑分更高的CSPResNext50,
如果把堆叠的残差单元(resblock_body)看成整体的话，那么这个结构和Darknet53以及ResNet等的确差别不大，特别是resblock_body的num_blocks为【1，2，8，8，4】和darknet53一模一样。 差别如图所示

目标检测模型的Neck部分主要用来融合不同尺寸特征图的特征信息。常见的有MaskRCNN中使用的FPN等， 用到了SPP（Spatial pyramid pooling）+PAN(Path Aggregation Network），v4neck图的结构b
可见，随着人们追求检测器在COCO数据集上的MAP指标,Neck部分也是出了很多花里胡哨的结构

通常将YOLO HEAD(SPP+PAN图片上的橙色块）紧接在SPP+PAN后面。为了便于说明，这里我们根据SPP+PAN图片上的process1-5与三个YOLO HEAD ,对SPP+PAN+YOLO HEAD 部分进行解析。

下面使用v4结构图
process1接受CSPDarknet53最终的输出，返回变量y19
process2将上述的y19进行上采样至大小38x38，然后再和CSPDarknet53的204层输出进行堆叠，最后通过一系列DarknetConv2D_BN_Leaky模块，获得特征图y38
process3的代码接受y_38上采样后的特征图 y38_upsample以及darknet网络的第131层输出作为输入，从而获得特征图y_38
YOLO HEAD 1紧接在process3之后，代码中使用简单的5+2层卷积层对上面的y76进行输出。其实这里的卷积层就是图中橙色区域YOLO HEAD1 ,在后面的y38_output和y19_output的输出过程中仍能够看到。该网络最后使用1x1卷积输出最大的一张特征图y76_output，维度为(76,76,num_anchor*(num_classes+5))。对应结构图中最大的输出特征图（最右边的淡蓝色特征图）。
process4这一步骤比较关键，PAN和FPN的差异在于，FPN是自顶向下的特征融合，PAN在FPN的基础上，多了个自底向上的特征融合。具体自底向上的特征融合，就是process4完成的，可以看到该步骤先将y76下采样至38x38大小，再和y38堆叠，作为YOLO HEAD2的输入。
YOLO HEAD 2类似于YOLO HEAD 1,YOLO HEAD2也进行一系列卷积运算，获得维度大小为(38,38,num_anchor*(num_classes+5))的输出y38_output 见yolohead2
Process5和process4进程类似，不多赘述。后面接上YOLO HEAD 3。
YOLO HEAD 3 和YOLO HEAD 1以及YOLO HEAD 2定义几乎类似，YOLO HEAD 3输出为(19,19,num_anchor*(num_classes+5)）的特征图y19_output。


MSE存在的一些问题：MSE损失函数将检测框中心点坐标和宽高等信息作为独立的变量对待的，但是实际上他们之间是有关系的。从直观上来说，框的中心点和宽高的确存在着一定的关系。所以解决方法是使用IOU损失代替MSE损失。
（1）IOU损失
（2）GIOU损失
（3）DIOU损失
（4）CIOU损失

该CIOU函数定义被用在求解总损失函数上了，我们知道YOLO V3的损失函数主要分为三部分，分别为：

（1）bounding box regression损失
（2）置信度损失
（3）分类损失
YOLO V4相较于YOLO V3,只在bounding box regression做了创新，用CIOU代替了MSE，其他两个部分没有做实质改变。最后对上述的三个损失取个平均即可

YOLOv4的特点是集大成者，俗称堆料。但最终达到这么高的性能，一定是不断尝试、不断堆料、不断调参的结果，给作者点赞。下面看看堆了哪些料：
1
Weighted-Residual-Connections (WRC)可以更好更快的结合不同层传递过来的残差，虽然增加了一些计算量，但是当网络层数从100+增加到1000+时，网络效果更好，收敛速度更快
2
Cross-Stage-Partial-connections
加强CNN的学习能力。
消除计算瓶颈。
降低内存成本。
3
Cross mini-Batch Normalization (CmBN)
CmBN代表CBN改进的版本。它只收集了一个批次中的mini-batches之间的统计数据
4
Self-adversarial-training (SAT)
自适应对抗训练（SAT）也表示了一个新的数据增广的技巧，它在前后两阶段上进行操作。在第一阶段，神经网络代替原始的图片而非网络的权重。用这种方式，神经网络自己进行对抗训练，代替原始的图片去创建图片中此处没有期望物体的描述。在第二阶段，神经网络使用常规的方法进行训练，在修改之后的图片上进行检测物体。
作者将SAM的spatial-wise注意力变成了point-wise注意力机制，然后将PAN中的shortcut连接变成了concatenation连接，正如图5和图6所表示的那样。

CIoU loss等等
用于backbone的BoF：CutMix和Mosaic数据增强，DropBlock正则化，Class label smoothing
用于backbone的BoS：Mish激活函数，CSP，MiWRC
用于检测器的BoF：CIoU-loss，CmBN，DropBlock正则化，Mosaic数据增强，Self-Adversarial 训练，消除网格敏感性，对单个ground-truth使用多个anchor，Cosine annealing scheduler，最佳超参数，Random training shapes
用于检测器的Bos：Mish激活函数，SPP，SAM，PAN，DIoU-NMS

主要贡献：提出了一种高效而强大的目标检测模型。它使每个人都可以使用1080 Ti或2080 Ti GPU 训练超快速和准确的目标检测器
验证了SOTA的Bag-of Freebies 和Bag-of-Specials方法的影响
改进了SOTA的方法，使它们更有效，更适合单GPU训练，包括CBN [89]，PAN [49]，SAM [85]等。文章将目前主流的目标检测器框架进行拆分：input、backbone、neck 和 head.

这里是简化版的v4
'''


class YOLOv4(nn.Module):
    def __init__(self, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.50,
                 anchor_size=None, pretrained=False, hr=False):
        super(YOLOv4, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = [8, 16, 32]  # 多stride，分成
        self.anchor_size = torch.Tensor(anchor_size).view(3, len(anchor_size) // 3, 2)
        self.num_anchors = len(anchor_size) // 3  # self.anchor_size.size(1)

        # 创建网格
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = create_gridv3(input_size, strides=self.stride,
                                                                                num_anchors=self.num_anchors,
                                                                                anchor_size=self.anchor_size)

        self.base = CSPDarknet()  # 对应v4结构图中CSPDarknet53
        if pretrained:
            path_to_dir = os.path.dirname(os.path.abspath(__file__))
            print('Loading the cspnet')
            if hr:
                self.base.load_state_dict(
                    torch.load(path_to_dir + '/weights/cspdarknet53/cspdarknet53_hr_76.9.pth'), strict=False)
            else:
                print('Loading the cspdarknet53 ...')
                self.base.load_state_dict(
                    torch.load(path_to_dir + '/weights/cspdarknet53/cspdarknet53_75.7.pth'), strict=False)

        # SPP网络用在YOLOv4中的目的是增加网络的感受野。实现是对layer107进行 5 × 5  、 9 × 9 、 13 × 13 的最大池化，分别得到layer 108，layer 110和layer 112，完成池化后，将layer 107，layer 108，layer 110和layer 112进行concatenete，连接成一个特征图layer 114并通过 1 × 1 1 \times 11×1降维到512个通道。
        # 将 self.base 输出的结果 输入spp
        self.spp = nn.Sequential(
            Conv(1024, 512, k=1),
            SPP(),  # 将 x 和 三个 maxpool 的结果 concat在一起
            CSPBottleneck(512 * 4, 1024, n=3, shortcut=False)
        )

        # 对spp输出的结果进行处理  1x1卷积->2X上采样->CSP瓶颈层
        self.head_conv_0 = Conv(1024, 512, k=1)
        self.head_upsample_0 = nn.Upsample(scale_factor=2)
        self.head_csp_0 = CSPBottleneck(512 * 2, 512, n=3, shortcut=False)  # 下面所有的CSPBottleneck的n都是3

        self.head_conv_1 = Conv(512, 256, 1)
        self.head_csp_1 = CSPBottleneck(256 * 2, 256, n=3, shortcut=False)
        self.head_upsample_1 = nn.Upsample(scale_factor=2)

        self.head_conv_2 = Conv(256, 256, k=3, p=1, s=2)
        self.head_csp_2 = CSPBottleneck(256 * 2, 512, n=3, shortcut=False)

        self.head_conv_3 = Conv(512, 512, k=3, p=1, s=2)
        self.head_csp_3 = CSPBottleneck(512 * 2, 1024, n=3, shortcut=False)

        self.head_det_1 = nn.Conv2d(256, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_2 = nn.Conv2d(512, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_3 = nn.Conv2d(1024, self.num_anchors * (1 + self.num_classes + 4), 1)

    def set_grid(self, x):
        self.input_size = x
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = create_gridv3(x, self.stride, self.num_anchors,
                                                                                self.anchor_size)

    def forward(self, x, target=None, eva=False):
        C_3, C_4, C_5 = self.base(x)

        # spp neck
        C_5 = self.spp(C_5)

        # FPN+PAN
        y0 = self.head_conv_0(C_5)
        y0out = self.head_upsample_0(y0)
        y0out = torch.cat([y0out, C_4], 1)
        y0out = self.head_csp_0(y0out)

        y1 = self.head_conv_1(y0out)
        y1out = self.head_upsample_1(y1)
        y1out = torch.cat([y1out, C_3], 1)
        y1out = self.head_csp_1(y1out)

        y2 = self.head_conv_2(y1out)
        y2out = torch.cat([y2, y1], 1)
        y2out = self.head_csp_2(y2out)

        y3 = self.head_conv_3(y2out)
        y3out = torch.cat([y3, y0], 1)
        y3out = self.head_csp_3(y3out)

        pred_s = self.head_det_1(y1out)  # 对small size进行处理
        pred_m = self.head_det_2(y2out)  # 对medium size进行处理
        pred_l = self.head_det_3(y3out)  # 对large size进行处理

        preds = [pred_s, pred_m, pred_l]
        total_conf_pred = []
        total_cls_pred = []
        total_txtytwth_pred = []
        b = hw = 0  #

        for pred in preds:  # 依次从小的感受野（感受区域较小，但是数量更多）到大
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
        
        # 与yolov3不同的是，location_loss使用了ciou。ciou在iou的基础上考虑了边框的重合度、中心距离和宽高比的尺度信息。
        # 此处先考虑一下如何来计算Loss值呢, 首先Loss是由目标分类(是否有目标)Loss+类别分类(具体子类别)Loss+中心偏移(x,y)Loss+宽高(w,h)Loss共同组成
        # 然后来分析一下各个Loss如何计算呢
        # 1:目标分类(是否有目标)Loss: 想计算此Loss那么需要知道哪些预测box中是有目标的, 哪些预测box中是没有目标的,
        # --> 标准为:此box与任一真实box(即labels中标注的box)的IOU均小于设定阈值则认作为无目标, 若此box负责某一真实box的预测,即与某一真实box有最大IOU/CIOU等, 则认作为有目标
        # 2: 其余的Loss均是针对负责预测的box来计算的, 即与某一真实box有最大IOU/CIOU等的那个预测的box
        # 然后让我们进入build_target方法中看具体的构造方式
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
            xyxy_pred = total_txtytwth_pred.view(b, hw, self.num_anchors, 4)
            with torch.no_grad():
                # batch size = 1
                # 测试时，笔者默认batch是1，
                # 因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W*num_anchor, 1] -> [H*W*num_anchor, 1]
                conf_pred = nn.Sigmoid()(total_conf_pred)[0]
                bboxes = torch.clamp(
                    (decode_boxes(xyxy_pred, self.all_anchors_wh, self.grid_cell) / self.input_size)[0], 0., 1.)
                scores = self.softmax(total_cls_pred[0, :, :]) * conf_pred

                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                bboxes, scores, cls_inds = postprocess(bboxes, scores, self.conf_thresh, self.num_classes,
                                                       self.nms_thresh)
                return bboxes, scores, cls_inds
