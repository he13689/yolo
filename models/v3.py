import torch
import torch.nn as nn
from models.basemodel import *
import numpy as np
from config import *
from utils.lossUtils import loss
from utils.moduleUtils import Conv, reorg_layer
from utils.utils import decode_boxes, iou_score, postprocess, create_gridv3
from models.basemodel.d19 import DarkNet_53

'''
YOLOv3 的先验检测（Prior detection）系统将分类器或定位器重新用于执行检测任务。
他们将模型应用于图像的多个位置和尺度。而那些评分较高的区域就可以视为检测结果。
此外，相对于其它目标检测方法，我们使用了完全不同的方法。
我们将一个单神经网络应用于整张图像，该网络将图像划分为不同的区域，因而预测每一块区域的边界框和概率，
这些边界框会通过预测的概率加权。我们的模型相比于基于分类器的系统有一些优势。
它在测试时会查看整个图像，所以它的预测利用了图像中的全局信息。与需要数千张单一目标图像的 R-CNN 不同，
它通过单一网络评估进行预测。这令 YOLOv3 非常快，一般它比 R-CNN 快 1000 倍、比 Fast R-CNN 快 100 倍。

此结构主要由75个卷积层构成，卷积层对于分析物体特征最为有效。由于没有使用全连接层，该网络可以对应任意大小的输入图像。
池化层也没有出现在YOLOv3当中，取而代之的是将卷积层的stride设为2来达到下采样的效果，同时将尺度不变特征传送到下一层。
YOLOv3中还使用了类似ResNet和FPN网络的结构，这两个结构对于提高检测精度也是大有裨益

3条预测支路采用的也是全卷积的结构，其中最后一个卷积层的卷积核个数是255，是针对COCO数据集的80类：
3*(80+4+1)=255，
3表示一个grid cell包含3个bounding box，4表示框的4个坐标信息，1表示objectness score。

yolov3是在训练的数据集上聚类产生prior boxes的一系列宽高(是在图像416x416的坐标系里)，默认9种。
YOLOV3思想理论是将输入图像分成SxS个格子（有三处进行检测，
分别是在52x52, 26x26, 13x13的feature map上，即S会分别为52,26,13）
若某个物体Ground truth的中心位置的坐标落入到某个格子，那么这个格子就负责检测中心落在该栅格中的物体。
三次检测，每次对应的感受野不同，32倍降采样的感受野最大（13x13），适合检测大的目标，
每个cell的三个anchor boxes为(116 ,90),(156 ,198)，(373 ,326)。
16倍（26x26）适合一般大小的物体，anchor boxes为(30,61)， (62,45)，(59,119)。
8倍的感受野最小（52x52），适合检测小目标，因此anchor boxes为(10,13)，(16,30)，(33,23)。
所以当输入为416×416时，实际总共有(52×52+26×26+13×13)×3=10647个proposal boxes。

yolov3是直接对你的训练样本进行k-means聚类，由训练样本得来的先验框（anchor boxes），也就是对样本聚类的结果。
通常一幅图像包含各种不同的物体，并且有大有小。比较理想的是一次就可以将所有大小的物体同时检测出来。因此，网络必须具备能够“看到”不同大小的物体的能力。并且网络越深，特征图就会越小，所以越往后小的物体也就越难检测出来。SSD中的做法是，在不同深度的feature map获得后，直接进行目标检测，这样小的物体会在相对较大的feature map中被检测出来，而大的物体会在相对较小的feature map被检测出来，从而达到对应不同scale的物体的目的。
然而在实际的feature map中，深度不同所对应的feature map包含的信息就不是绝对相同的。举例说明，随着网络深度的加深，浅层的feature map中主要包含低级的信息（物体边缘，颜色，初级位置信息等），深层的feature map中包含高等信息（例如物体的语义信息：狗，猫，汽车等等）。因此在不同级别的feature map中进行检测，听起来好像可以对应不同的scale，但是实际上精度并没有期待的那么高。
在YOLOv3中，这一点是通过采用FPN结构来提高对应多重scale的精度的。 关于fpn图，见不同feature map处理方式.jpg

如上图所示，对于多重scale，目前主要有以下几种主流方法。

(a) Featurized image pyramid: 这种方法最直观。首先对于一幅图像建立图像金字塔，不同级别的金字塔图像被输入到对应的网络当中，用于不同scale物体的检测。但这样做的结果就是每个级别的金字塔都需要进行一次处理，速度很慢。
(b) Single feature map: 检测只在最后一个feature map阶段进行，这个结构无法检测不同大小的物体。
(c) Pyramidal feature hierarchy: 对不同深度的feature map分别进行目标检测。SSD中采用的便是这样的结构。每一个feature map获得的信息仅来源于之前的层，之后的层的特征信息无法获取并加以利用。
(d) Feature Pyramid Network 与(c)很接近，但有一点不同的是，当前层的feature map会对未来层的feature map进行上采样，并加以利用。这是一个有跨越性的设计。因为有了这样一个结构，当前的feature map就可以获得“未来”层的信息，这样的话低阶特征与高阶特征就有机融合起来了，提升检测精度。

YOLOv3中使用了ResNet结构（对应着在上面的YOLOv3结构图中的Residual Block）。
Softmax层被替换为一个1x1的卷积层+logistic激活函数的结构。使用softmax层的时候其实已经假设每个输出仅对应某一个单个的class，但是在某些class存在重叠情况（例如woman和person）的数据集中，使用softmax就不能使网络对数据进行很好的拟合。

其实图像在输入之前是按照图像的长边缩放为416，短边根据比例缩放(图像不会变形扭曲)，然后再对短边的两侧填充至416，这样就保证了输入图像是416*416的。

注意点：loss计算时 anchor box与ground truth的匹配。
为啥需要匹配呢？你是监督学习，那得知道网络预测的结果是啥呀？这样才能逼近真实的label，反过来就是我现在让他们匹配，给他分配好label，后面就让网络一直这样学习，最后就是以假乱真了，输出的结果无线接近正确结果了。​yolov3的输出prediction的shape为(num_samples, self.num_anchors*(self.num_classes + 5), grid_size, grid_size)，为了计算loss，转换为(num_samples, self.num_anchors, grid_size, grid_size, 5+self.num_classes ), 其中self.num_anchors为3, 总共这么多boxes，哪些框可能有目标呢，而且一个cell对应有三个anchor boxes，究竟选择哪个anchor去匹配ground truth？
将每个锚框（anchor boxes）视为一个训练样本，需要标记每个anchor box的标签，即类别标签和偏移量。所以我们只需要考虑有目标的anchor boxes，哪些有目标呢？ground truth的中心落在哪个cell，那对应这三个anchor boxes就有，所以计算ground truth与anchor boxeses的IOU（bbox_wh_iou(计算Gw，Gh与Pw,Ph的IOU)），其实只需要选取重叠度最高的anchor box就行，再将三个anchores通过torch.stack后max(0)下就知道选择三个中的哪个了，将这种方式匹配到的boxes视为有目标的box。

yolov3是直接对你的训练样本进行k-means聚类，由训练样本得来的先验框（anchor boxes），也就是对样本聚类的结果。
每种尺度预测3个box, anchor的设计方式仍然使用聚类,得到9个聚类中心,将其按照大小均分给3个尺度.
尺度1: 在基础网络之后添加一些卷积层再输出box信息.
尺度2: 从尺度1中的倒数第二层的卷积层上采样(x2)再与最后一个16x16大小的特征图相加,再次通过多个卷积后输出box信息.相比尺度1变大两倍.
尺度3: 与尺度2类似,使用了32x32大小的特征图.


改进之处
多尺度预测 （引入FPN）。
更好的基础分类网络（darknet-53, 类似于ResNet引入残差结构）。
分类器不在使用Softmax，分类损失采用binary cross-entropy loss（二分类交叉损失熵）
YOLOv3不使用Softmax对每个框进行分类，主要考虑因素有两个：

Softmax使得每个框分配一个类别（score最大的一个），而对于Open Images这种数据集，目标可能有重叠的类别标签，因此Softmax不适用于多标签分类。
Softmax可被独立的多个logistic分类器替代，且准确率不会下降。
分类损失采用binary cross-entropy loss。
详情见 https://zhuanlan.zhihu.com/p/362761373
'''


class YOLOv3(nn.Module):  # v3 based on d53
    def __init__(self, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.50,
                 anchor_size=None):
        super(YOLOv3, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = [8, 16, 32] # 多stride，分成  如果它对于不同尺寸的输入进行了压缩，但是base网络输出并不改变，就会导致维度不同
        self.anchor_size = torch.Tensor(anchor_size).view(3, len(anchor_size) // 3, 2)
        self.num_anchors = self.anchor_size.size(1) # 2

        # 创建网格  与v2不同之处在于它对于多个规格的stride 创建多个grid  torch.Size([1, 3024, 1, 2])
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = create_gridv3(input_size, strides=self.stride,
                                                                                num_anchors=self.num_anchors,
                                                                                anchor_size=self.anchor_size)
        # backbone darknet-53
        self.base = DarkNet_53()  # 输出维度1024

        # s = 32 对应yolov3中y1支路的DBL5
        self.conv_set_3 = nn.Sequential(
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1)
        )
        # 用于传给y2支路
        self.conv_1x1_3 = Conv(512, 256, k=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # y1支路输出层 DBL和Conv
        self.extra_conv_3 = Conv(512, 1024, k=3, p=1)
        self.pred_3 = nn.Conv2d(1024, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv(768, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1)
        )
        self.conv_1x1_2 = Conv(256, 128, k=1)
        self.extra_conv_2 = Conv(256, 512, k=3, p=1)
        self.pred_2 = nn.Conv2d(512, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv(384, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1)
        )
        self.extra_conv_1 = Conv(128, 256, k=3, p=1)
        self.pred_1 = nn.Conv2d(256, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

    def set_grid(self, x):
        self.input_size = x
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = create_gridv3(x, self.stride, self.num_anchors,
                                                                                self.anchor_size)

    def forward(self, x, target=None, eva=False):
        C_3, C_4, C_5 = self.base(x)

        # y1支路流程 最简单
        y1 = self.conv_set_3(C_5)
        # y1支路最后的DBL+conv
        y1out = self.extra_conv_3(y1)
        y1out = self.pred_3(y1out)

        y1_extra = self.conv_1x1_3(y1)
        y1_extra = self.upsample3(y1_extra)

        y2 = torch.cat([C_4, y1_extra], 1)
        y2 = self.conv_set_2(y2)
        y2out = self.extra_conv_2(y2)
        y2out = self.pred_2(y2out)

        y2_extra = self.conv_1x1_2(y2)
        y2_extra = self.upsample2(y2_extra)

        y3 = torch.cat([C_3, y2_extra], 1)
        y3 = self.conv_set_1(y3)
        y3out = self.extra_conv_1(y3)
        y3out = self.pred_1(y3out)
        
#        print(y3out.size(), y2out.size(), y1out.size()) # torch.Size([1, 75, 40, 40]) torch.Size([1, 75, 20, 20]) torch.Size([1, 75, 10, 10])
        preds = [y3out, y2out, y1out]

        # 记录下每个scale[y3out, y2out, y1out]下 每个anchor的conf cls xywh
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
            bb_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
#            print(bb_pred.size())
            '''
            当 图像大小为640时
            torch.Size([1, 6400, 12])
            torch.Size([1, 1600, 12])
            torch.Size([1, 400, 12])
            
            cat 后 pred 的 channel 1 维度是8400
            '''

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
#            print(bb_pred.size()) # torch.Size([1, 8400, 3, 4])
            
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
            
            # (total_conf_pred, total_cls_pred, xyxy_pred, iou_pred, target) ： torch.Size([1, 6300, 1]) torch.Size([1, 6300, 20]) torch.Size([1, 6300, 4]) torch.Size([1, 6300, 1]) torch.Size([1, 6300, 8])
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
