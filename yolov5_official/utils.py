import math
import random
import re

import cv2
import torch.nn as nn
import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm


#
#
# def list_of_groups(list_info, per_list_len):
#     '''
#     :param list_info:   列表
#     :param per_list_len:  每个小列表的长度
#     :return:
#     '''
#     list_of_group = zip(*(iter(list_info),) * per_list_len)
#     end_list = [list(i) for i in list_of_group]  # i is a tuple
#     count = len(list_info) % per_list_len
#     end_list.append(list_info[-count:]) if count != 0 else end_list
#     return end_list


# lr梯度更新策略
def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def detection_collate(batch):
    """
    自定义的collate fn， 用于处理批图像中不同数量的关联物体标注

    输入是一个元组，里面包括image tensor以及标注的列表

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


# 这是一个数据采集function
def v5_collate_fn(batch):
    img, label, path, shapes = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0), path, shapes


# 非极大值抑制
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    print('nms is departed, please use diou nms! ')

    nc = prediction.shape[2] - 5  # 类别数量
    xc = prediction[..., 4] > conf_thres  # 候选框

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = max_det  # 最多检测多少个nms box在每张image上
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling  # 用于自动标签 现在基本没用
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.Tensor(classes)).cuda().any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # 批 NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output


def intersect(box_a, box_b):
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard_diou(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    x1 = ((box_a[:, :, 2] + box_a[:, :, 0]) / 2).unsqueeze(2).expand_as(inter)
    y1 = ((box_a[:, :, 3] + box_a[:, :, 1]) / 2).unsqueeze(2).expand_as(inter)
    x2 = ((box_b[:, :, 2] + box_b[:, :, 0]) / 2).unsqueeze(1).expand_as(inter)
    y2 = ((box_b[:, :, 3] + box_b[:, :, 1]) / 2).unsqueeze(1).expand_as(inter)

    t1 = box_a[:, :, 1].unsqueeze(2).expand_as(inter)
    b1 = box_a[:, :, 3].unsqueeze(2).expand_as(inter)
    l1 = box_a[:, :, 0].unsqueeze(2).expand_as(inter)
    r1 = box_a[:, :, 2].unsqueeze(2).expand_as(inter)

    t2 = box_b[:, :, 1].unsqueeze(1).expand_as(inter)
    b2 = box_b[:, :, 3].unsqueeze(1).expand_as(inter)
    l2 = box_b[:, :, 0].unsqueeze(1).expand_as(inter)
    r2 = box_b[:, :, 2].unsqueeze(1).expand_as(inter)

    cr = torch.max(r1, r2)
    cl = torch.min(l1, l2)
    ct = torch.min(t1, t2)
    cb = torch.max(b1, b2)
    D = (((x2 - x1) ** 2 + (y2 - y1) ** 2) / ((cr - cl) ** 2 + (cb - ct) ** 2 + 1e-7))
    out = inter / area_a if iscrowd else inter / (union + 1e-7) - D ** 0.7
    return out if use_batch else out.squeeze(0)


def jaccard(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / (union + 0.0000001)

    return out if use_batch else out.squeeze(0)


def bbox_iou2(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    """iou giou diou ciou
    Args:
        box1: 预测框
        box2: 真实框
        x1y1x2y2: False
    Returns:
        box1和box2的IoU/GIoU/DIoU/CIoU
    """
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()  # 转置 ？？？

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2  # b1左上角和右下角的x坐标
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2  # b1左下角和右下角的y坐标
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2  # b2左上角和右下角的x坐标
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2  # b2左下角和右下角的y坐标

    # Intersection area  tensor.clamp(0): 将矩阵中小于0的元数变成0
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter  # 1e-16: 防止分母为0

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # return GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


def non_max_suppression2(prediction, conf_thres=0.25, nms_thres=0.6, multi_cls=True, method='diou_nms'):
    """
        移除小于conf_thres的框
        param:
             prediction: [batch, num_anchors, (x+y+w+h+conf+num_classes)]  3个anchor set的预测结果的集合
             conf_thres: 先进行一轮筛选，将分数过低的预测框（<conf_thres）删除（分数置0）
             nms_thres: iou阈值, 如果其余预测框与target的iou>iou_thres, 就将那个预测框置0
             multi_label: 是否是多标签
             method: nms方法
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, conf, class)
    """
    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) 宽度和高度的大小范围 [min_wh, max_wh]
    output = [None] * len(prediction)  # batch_size个output  存放最终筛选后的预测框结果
    for image_i, pred in enumerate(prediction):
        # 开始  pred = [12096, 25]
        # 第一层过滤   根据conf_thres虑除背景目标(conf<conf_thres的目标)
        pred = pred[pred[:, 4] > conf_thres]  # pred = [45, 25]

        # 第二层过滤   虑除超小anchor标和超大anchor  x=[45, 25]
        pred = pred[(pred[:, 2:4] > min_wh).all(1) & (pred[:, 2:4] < max_wh).all(1)]

        # 经过前两层过滤后如果该feature map没有目标框了，就结束这轮直接进行下一个feature map
        if len(pred) == 0:
            continue

        # 计算 score
        pred[..., 5:] *= pred[..., 4:5]  # score = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(pred[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_cls or conf_thres < 0.01:
            # 第三轮过滤: 针对每个类别score(obj_conf * cls_conf) > conf_thres
            # 这里一个框是有可能有多个物体的，所以要筛选
            # nonzero: 获得矩阵中的非0数据的下标  t(): 将矩阵拆开
            # i: 下标   j: 类别   shape=43  过滤了两个score太低的
            i, j = (pred[:, 5:] > conf_thres).nonzero(as_tuple=False).t()
            # pred = [43, xyxy+conf+class]
            pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = pred[:, 5:].max(1)  # 一个类别直接取分数最大类的即可
            pred = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # 第三轮过滤后如果该feature map没有目标框了，就结束这轮直接进行下一个feature map
        if len(pred) == 0:
            continue
        # 第四轮过滤  这轮可有可无，一般没什么用
        # pred = pred[torch.isfinite(pred).all(1)]

        # 降序排列 为NMS做准备
        pred = pred[pred[:, 4].argsort(descending=True)]

        # Batched NMS
        # Batched NMS推理时间：0.054
        if method == 'hard_nms_batch':  # 普通的(hard)nms: 官方实现(c函数库),可支持gpu,但支持多类别输入
            # batched_nms：参数1 [43, xyxy]  参数2 [43, score]  参数3 [43, class]  参数4 [43, nms_thres]
            output[image_i] = pred[torchvision.ops.boxes.batched_nms(pred[:, :4], pred[:, 4], pred[:, 5], nms_thres)]
            # print("hard_nms_batch")
            continue

        # All other NMS methods
        det_max = []  # 存放分数最高的框 即target
        cls = pred[:, -1]
        for c in cls.unique():  # 对所有的种类(不重复)
            dc = pred[cls == c]  # dc: 选出pred中所有类别是c的结果
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 500:  # 密集性 主要考虑到NMS是一个速度慢的算法（O(n^2)）,预测框太多,算法的效率太慢 所以这里筛选一下（最多500个预测框）
                dc = dc[:500]  # limit to first 500 boxes: https://github.com/ultralytics/yolov3/issues/117

            # 推理时间：0.001
            if method == 'hard_nms':  # 普通的(hard)nms: 只支持单类别输入
                det_max.append(dc[torchvision.ops.boxes.nms(dc[:, :4], dc[:, 4], nms_thres)])

            # 推理时间：0.00299 是官方写的3倍
            elif method == 'hard_nms_myself':  # Hard NMS 自己写的 只支持单类别输入
                while dc.shape[0]:  # dc.shape[0]: 当前class的预测框数量
                    det_max.append(dc[:1])  # 让score最大的一个预测框(排序后的第一个)为target
                    if len(dc) == 1:  # 出口 dc中只剩下一个框时，break
                        break
                    # dc[0] ：target     dc[1:] ：其他预测框
                    diou = bbox_iou2(dc[0], dc[1:])  # 计算 diou
                    dc = dc[1:][diou < nms_thres]  # remove dious > threshold

            # 在hard-nms的逻辑基础上，增加是否为单独框的限制，删除没有重叠框的框（减少误检）。
            elif method == 'and':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou2(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:  # 删除没有重叠框的框/iou小于0.5的框（减少误检）
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            # 在hard-nms的基础上，增加保留框位置平滑策略（重叠框位置信息求解平均值），使框的位置更加精确。
            elif method == 'merge':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou2(dc[0], dc) > nms_thres  # i = True/False的集合
                    weights = dc[i, 4:5]  # 根据i，保留所有True
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()  # 重叠框位置信息求解平均值
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            # 推理时间：0.0030s
            elif method == 'soft_nms':  # soft-NMS      https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    # if len(dc) == 1:  这是U版的源码 我做了个小改动
                    #     det_max.append(dc)
                    #     break
                    # det_max.append(dc[:1])
                    det_max.append(dc[:1])  # 保存dc的第一行  target
                    if len(dc) == 1:
                        break
                    iou = bbox_iou2(dc[0], dc[1:])  # 计算target与其他框的iou

                    # 这里和上面的直接置0不同，置0不需要管维度
                    dc = dc[1:]  # dc=target往后的所有预测框
                    # dc必须不包括target及其前的预测框，因为还要和值相乘, 维度上必须相同
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # 得分衰减
                    dc = dc[dc[:, 4] > conf_thres]

            # 推理时间：0.00299
            elif method == 'diou_nms':  # DIoU NMS  https://arxiv.org/pdf/1911.08287.pdf
                while dc.shape[0]:  # dc.shape[0]: 当前class的预测框数量
                    det_max.append(dc[:1])  # 让score最大的一个预测框(排序后的第一个)为target
                    if len(dc) == 1:  # 出口 dc中只剩下一个框时，break
                        break
                    # dc[0] ：target     dc[1:] ：其他预测框
                    diou = bbox_iou2(dc[0], dc[1:], DIoU=True)  # 计算 diou
                    dc = dc[1:][diou < nms_thres]  # remove dious > threshold  保留True 删去False

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate  因为之前是append进det_max的
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # 排序

    return output


def diou_non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, max_box=1500, classes=None, agnostic=False,
                             use_gpu=False, max_det=300, labels=None, multi_label=True):
    """
    对结果使用Diou nms
    返回结果和原来NMS的结果格式一样
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # 设置
    min_wh, max_wh = 2, 4096  # 以像素为单位的 最大最小box框
    multi_label = nc > 1  # 是否有多label， 即非single cls
    output = [None] * prediction.shape[0]
    # pred1.size()=[batch, max_box, 6] denotes boxes without offset by class
    pred1 = (prediction < -1).float()[:, :max_box, :6]
    pred2 = pred1[:, :, :4] + 0  # pred2 denotes boxes with offset by class
    batch_size = prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        x = x[x[:, 4].argsort(descending=True)]
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes

        boxes = (x[:, :4].clone() + c.view(-1, 1) * max_wh)[:max_box]  # boxes (offset by class), scores
        pred2[xi, :] = torch.cat((boxes, pred2[xi, :]), 0)[:max_box]  # If less than max_box, padding 0.
        pred1[xi, :] = torch.cat((x[:max_box], pred1[xi, :]), 0)[:max_box]

    # Batch mode Cluster-Weighted NMS

    iou = jaccard_diou(pred2, pred2).triu_(diagonal=1)  # switch to 'jaccard_diou' function for using Cluster-DIoU-NMS
    B = iou
    for i in range(200):
        A = B
        maxA = A.max(dim=1)[0]
        E = (maxA < iou_thres).float().unsqueeze(2).expand_as(A)
        B = iou.mul(E)
        if A.equal(B) == True:
            break
    keep = (maxA <= iou_thres)
    if use_gpu:
        weights = (B * (B > 0.8) + torch.eye(max_box).cuda().expand(batch_size, max_box, max_box)) * (
            pred1[:, :, 4].reshape((batch_size, 1, max_box)))
    else:
        weights = (B * (B > 0.8) + torch.eye(max_box).expand(batch_size, max_box, max_box)) * (
            pred1[:, :, 4].reshape((batch_size, 1, max_box)))
    pred1[:, :, :4] = torch.matmul(weights, pred1[:, :, :4]) / weights.sum(2, keepdim=True)  # weighted coordinates

    for jj in range(batch_size):
        output[jj] = pred1[jj][keep[jj]]

    return output


def nms(dets, scores, threshold):
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax

    areas = (x2 - x1) * (y2 - y1)  # the size of bbox
    order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order

    keep = []  # store the final bounding boxes
    while order.size > 0:
        i = order[0]  # the index of the bbox with highest confidence
        keep.append(i)  # save it to keep
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ComputeLoss:
    def __init__(self, model, hyp, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.autobalance = autobalance
        # 选择计算 loss 所用的函数 pos_weight指正样本的weight
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([hyp['cls_pw']]))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([hyp['obj_pw']]))

        # 用于对 正负标签 进行smooth
        self.stl, self.sfl = smooth_BCE(eps=hyp.get('label_smoothing', 0.0))

        if hyp['fl_gamma'] > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, hyp['fl_gamma']), FocalLoss(BCEobj, hyp['fl_gamma'])

        detect = model.model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(detect.nl, [4.0, 1.0, 0.25, 0.06,
                                                            .02])  # 如果detect.nl=3， 那么[4.0, 1.0, 0.4]，也就是说有多少组anchor
        self.ssi = list(detect.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, hyp, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':  # 设置了'na', 'nc', 'nl', 'anchors' 从detect中
            setattr(self, k, getattr(detect, k))

    def __call__(self, pred, targets, use_gpu=False):
        '''
        输入为
        targets : torch.Size([58, 6])
        
        pred : List 3(代表不同大小anchor set和特征图输出的结果), torch.Size([4（图片4张，bs=4）, 3（每个set的三个不同形状anchor）, 80（x有80个坐标）, 80（y有80个坐标）, 85（分类、pos、conf）]) torch.Size([4, 3, 40, 40, 85]) torch.Size([4, 3, 20, 20, 85])
        其中bs=4， 80是特征图size， anchor num是3， 85中由80个分类，1个conf以及4个大小坐标组成
        '''
        # 定义三个loss 分辨是分类、box区域以及 obj（是否有目标）
        if use_gpu:
            lcls, lbox, lobj = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()
        else:
            lcls, lbox, lobj = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        # 使用build函数来建立target， 输出target的cls， box， indices以及anchor， 他们都是list ，长度为3, 其中他们表示对于一张图像， 任取一个target，它的类别、box和
        tcls, tbox, indices, anchors = self.build_targets(pred, targets)
        print('\r')
        # torch.Size([50, 6]) torch.Size([168]) torch.Size([168, 4]) List(3 * 4 * torch.Size([168])) torch.Size([168, 2])
        #        print(targets.shape, tcls[0].shape, tbox[0].shape, indices[0][0].shape, anchors[0].shape)
        #        print(tcls[0][0], tbox[0][0], indices[0][0][0], anchors[0][0])  tensor(22, device='cuda:0') tensor([ 0.0000,  0.0000,  3.9208,  4.7879], device='cuda:0') tensor(1, device='cuda:0') tensor([ 1.2500,  1.6250], device='cuda:0')
        for i, p in enumerate(pred):
            b, a, gj, gi = indices[
                i]  # 每个indices 里面有四个tensor，分别是 iamge, anchor, 以及 grid 上的 xy 代表第b张image的第a个anchor，它的坐标是gj和gi
            #            print(p.shape, b.shape, a.shape, gj.shape, gi.shape)  # torch.Size([4, 3, 80, 80, 85]) torch.Size([447]) torch.Size([447]) torch.Size([447]) torch.Size([447])
            #            print(b[0], a[0], gj[0], gi[0]) # tensor(1, device='cuda:0') tensor(0, device='cuda:0') tensor(11.1508, device='cuda:0') tensor(72.8764, device='cuda:0')
            tobj = torch.zeros_like(p[..., 0])  # torch.Size([4, 3, 80, 80])
            n = b.shape[0]  # b 有数百个，表示这四张图上所有pred出的目标
            if n:  # 如果有检测出目标
                # gj, gi 都是long型的Tensor，它们作为index的话，比如0.8，那么相当于向下取整，返回索引=0的结果
                ps = p[b, a, gj, gi]  # 从pred中取出target中所对应位置的预测
                # 做一个regression 回归
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # 对比pred出的box和tbox，看看它们两个之间的差异  使用CIOU进行改进
                iou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss  计算ious loss

                # 目标 objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # classification 分类
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.sfl)  # 使用了soft label， 将false label 设置为sfl
                    t[range(n), tcls[i]] = self.stl  # 将正确label设置为stf
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE 计算分类的损失

            obji = self.BCEobj(p[..., 4], tobj)  # 是否包含目标
            lobj += obji * self.balance[i]  # obj loss

        #            if self.autobalance:
        #                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        #
        if self.autobalance:
            self.balance = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        # 将不同的loss来源乘上相应的权重， 比如box的权重就是0.05
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        # 将所有的loss 加在一起看
        loss = lbox + lobj + lcls
        # 多加入了一个 总loss
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, pred, targets, use_gpu=False):
        # 下面参数假设targets有39个
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []  # 返回的list
        gain = torch.ones(7).cuda() if use_gpu else torch.ones(7)  # normalized to gridspace gain 是[0,0,.....0] 增益值
        if use_gpu:
            ai = torch.arange(na).cuda().float().view(na, 1).repeat(1, nt)
        else:
            ai = torch.arange(na).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        #        print(ai.shape, targets.shape)  # torch.Size([3, 39]) torch.Size([39, 6])
        '''
        ai: when na = 3
        tensor([[ 0.,  0.,  0.,  0.,  0. nt个0.],
        [ 1.,  1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.,  2.]])
        '''

        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        g = 0.5
        if use_gpu:
            off = torch.Tensor([[0, 0],
                                [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                                ]).cuda().float() * g
        else:
            off = torch.Tensor([[0, 0],
                                [1, 0], [0, 1], [-1, 0], [0, -1],
                                ]).float() * g

        for i in range(self.nl):  # 根据layer数量 3
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # ori target shape:torch.Size([39, 6])
            # tensor([  1.,   1.,  80.,  80.,  80.,  80.,   1.], device='cuda:0') torch.Size([7]) torch.Size([3, 39, 7])
            #            print(gain, gain.shape,targets.shape)
            # Match targets to anchors
            t = targets * gain  # 目标乘上增益
            if nt:
                '''
                分析可知， targets是 image, class,  x y , w h 一共6维
                其中 
                image是index
                tensor([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,
                 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,
                 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
                 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  3.,  3.,  3.,  3.,
                 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,
                 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,
                 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,
                 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,
                 3.])
                
                class 是 80类中 分类结果
                tensor([ 45.,  45.,  50.,  45.,  79.,  79.,  79.,  41.,  23.,  23.,
                 22.,  22.,  22.,   0.,  50.,  35.,  35.,   0.,   0.,   0.,
                  0.,  58.,  75.,   0.,  56.,  56.,  60.,   0.,   0.,   0.,
                  0.,   0.,   0.,   0.,   0.,  41.,  56.,  56.,  60.,   0.,
                  0.,  41.,  41.,  41.,  22.,  13.,  77.,  56.,   0.,   9.,
                  9.,  24.,   7.,   7.,   0.,   0.,   0.,   0.,   0.,   0.,
                  0.,   0.,   0.,   0.,   0.,   9.,   9.,   9.,  24.,  26.,
                  0.,   7.,   7.,   7.,   9.,  26.,   0.,   9.,   9.,   9.,
                 26.,  26.,  58.,  60.,  43.,  44.,  55.,  73.,  13.,  13.,
                 25.,  25.,  26.,  13.,  13.,  60.,  25.])
                
                '''
                # 匹配
                r = t[:, :, 4:6] / anchors[:, None]  # 宽高比
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).t()
                l, m = ((gxi % 1. < g) & (gxi > 1.)).t()
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().t()  # image, class
            gxy = t[:, 2:4].long()  # grid xy
            gwh = t[:, 4:6].long()  # grid wh
            gij = (gxy - offsets.long())
            gi, gj = gij.t()  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        # 返回的indices是 image, class， 
        return tcls, tbox, indices, anch


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # 返回正负标签经过smooth后的target  正标签1.0 - 0.5 * eps  在一定程度上可以减少过拟合
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def labels_to_class_weights(labels, nc=80):
    # 得到一个类别权重 从训练标签中
    if labels[0] is None:
        return torch.Tensor()

    # 数据集中一共有多少个label
    labels = np.concatenate(labels, 0)  # label的shape = (866643, 5) [class xywh] 如果数据集是coco
    classes = labels[:, 0].astype(np.int)  # 取出所有 label 的类别
    weights = np.bincount(classes, minlength=nc)  # 每个类别出现的次数  进行一个统计，相当于投票统计  最少有nc个

    weights[weights == 0] = 1  # replace empty bins with 1  # 将没有出现的类也设置为1
    weights = 1 / weights  # 每个class有多少个targets  就等于出现频率
    weights /= weights.sum()  # 正则化
    return torch.from_numpy(weights)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    box1 = box1.float()
    box2 = box2.float()
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


# 评估最好的模型 给map不同的权重值，表示它们在评选最好模型时的比重
def fitness(x, w=None):
    # Model fitness as a weighted combination of metrics
    w = w or [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


# 根据论文， 将80 index 的val2014 转为91 index
def coco80_to_coco91_class():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def process_batch(predictions, labels, iouv):
    # Evaluate 1 batch of predictions
    correct = torch.zeros(predictions.shape[0], len(iouv), dtype=torch.bool, device=iouv.device)
    detected = []  # label indices
    tcls, pcls = labels[:, 0], predictions[:, 5]
    nl = labels.shape[0]  # number of labels
    for cls in torch.unique(tcls):
        ti = (cls == tcls).nonzero().view(-1)  # label indices
        pi = (cls == pcls).nonzero().view(-1)  # prediction indices
        if pi.shape[0]:  # find detections
            ious, i = box_iou(predictions[pi, 0:4], labels[ti, 1:5]).max(1)  # best ious, indices
            detected_set = set()
            for j in (ious > iouv[0]).nonzero():
                d = ti[i[j]]  # detected label
                if d.item() not in detected_set:
                    detected_set.add(d.item())
                    detected.append(d)  # append detections
                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                    if len(detected) == nl:  # all labels already located in image
                        break
    return correct


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def check_img_size(img_size, s=32, floor=0):
    # Verify img_size is a multiple of stride s
    new_size = max(make_divisible(img_size, int(s)), floor)  # ceil gs-multiple
    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print(f'thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, encoding='ascii', errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from yolov5_official.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans calculation
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    assert len(k) == n, print(f'ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)

    return print_results(k)


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    print('caculate auto anchor size  自适应计算锚框大小')
    # Check anchor fit to data, recompute if necessary
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)  # current anchors
    bpr, aat = metric(anchors)
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to improve anchors, please wait...')
        na = m.anchor_grid.numel() // 2  # number of anchors
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            print(f'ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)  # for inference
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print(f'New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print(f'Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


# 加载前23层的模型 适用于v5s
def load_pt(net, pretrained_model_src):
    pt = torch.load(pretrained_model_src)
    state = pt['model'].float().state_dict()
    net_state = net.state_dict()
    keys = state.keys()
    for s in state:
        # 不加载detect的参数
        if '24' in s:
            continue
        net_state.get(s).copy_(state.get(s))


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        truncated_normal_(m.weight, std=.01)
        if m.bias is not None:
            zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        zeros_(m.bias)
        ones_(m.weight)


# 自定义truncated_normal_函数
def truncated_normal_(tensor, mean=0, std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def zeros_(tensor):
    with torch.no_grad():
        tensor.data.fill_(0.0)
        return tensor


def ones_(tensor):
    with torch.no_grad():
        tensor.data.fill_(1.0)
        return tensor


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A

            self.transform = A.Compose([
                A.Blur(p=0.1),
                A.MedianBlur(p=0.1),
                A.ToGray(p=0.01)],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            print(e)

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


#  保存训练结果 图片形式
def save_box_img(image, targets, wid, hei, prefix='', names=''):
    image = image.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    cv2.imwrite(prefix + '_ori.jpg', image)
    img_ = cv2.imread(prefix + '_ori.jpg')
    for box in targets:
        x1, y1, w, h = box[2:]  # 这里的x y是中心坐标
        label = int(box[0])
        # print(xmin, ymin, xmax, ymax)
        x1 = x1 - w / 2
        y1 = y1 - h / 2

        x1 *= wid
        y1 *= hei
        x2 = x1 + w * wid
        y2 = y1 + h * hei
        cv2.rectangle(img_, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img_, names[label], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1)
    try:
        cv2.imwrite(prefix, img_)
    except:
        print('打印失败')


def mixup(im, labels, im2, labels2):
    r = np.random.beta(32.0, 32.0)
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels
