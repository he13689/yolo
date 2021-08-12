# 功能包括产生anchor锚点、计算iou，设置anchor、创建ground truth等功能
import numpy as np
import torch
import torch.nn as nn

import config
from config import *

'''
以下函数用于yolo 模型内部，对模型结果进行处理， 包括nms xywh_to_xxyy decode_boxes 等方法
'''


# 定义nms
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


# 将xywh坐标转成xxyy 原decode_xywh
def xywh_to_xxyy(xywh, grid_cell, all_anchor_wh, stride=32):
    """
        Input:
            txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
        Output:
            xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
    """
    B, HW, ab_n, _ = xywh.size()
    xy_pred = nn.Sigmoid()(xywh[:, :, :, :2]) + grid_cell.float()
    wh_pred = torch.exp(xywh[:, :, :, 2:]) * all_anchor_wh
    xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * stride

    return xywh_pred


def xywh_to_xxyyv3(xywh, grid_cell, all_anchor_wh, stride=32):
    """
        Input:
            txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
        Output:
            xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
    """
    B, HW, ab_n, _ = xywh.size()
    #    print(xywh[:, :, :, :2].size(), grid_cell.size())
    xy_pred = (nn.Sigmoid()(xywh[:, :, :, :2]) + grid_cell.float()) * stride
    wh_pred = torch.exp(xywh[:, :, :, 2:]) * all_anchor_wh
    xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4)

    return xywh_pred


def decode_boxes(xywh_pred, all_anchor_wh, grid_cell, requires_grad=False, v2=True):
    """
        Input:
            txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            grid_cell : grid 坐标系 下面的坐标
        Output:
            x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
    """
    if v2:
        # [H*W*anchor_n, 4]
        xywh_pred = xywh_to_xxyy(xywh_pred, grid_cell, all_anchor_wh)
    else:
        xywh_pred = xywh_to_xxyyv3(xywh_pred, grid_cell, all_anchor_wh)

    # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
    x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
    x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
    x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
    x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
    x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

    return x1y1x2y2_pred


def postprocess(bb, scores, conf_thresh, num_classes, nms_threshold):
    """
    bboxes: (HxW, 4), bsize = 1
    scores: (HxW, num_classes), bsize = 1
    """
    # 先找到最大置信度的分类
    cls_index = np.argmax(scores, axis=1)  # 找到score最高的 cls index 每个WH中最大的置信度的分类
    scores = scores[(np.arange(scores.shape[0]), cls_index)]  # 找到所有scores 重建scores只保留最好的index
    keep = np.where(scores >= conf_thresh)  # 找到大于置信度threshold的bb

    # 得到keep的bb，scores，cls index
    bb = bb[keep]
    scores = scores[keep]
    cls_index = cls_index[keep]

    # NMS 找到需要 keep的 index
    keep = np.zeros(len(bb), dtype=np.int)
    for i in range(num_classes):
        index = np.where(cls_index == i)[0]
        if len(index) == 0:
            continue
        cbb = bb[index]
        cscore = scores[index]
        ckeep = nms(cbb, cscore, nms_threshold)
        keep[index[ckeep]] = 1

    keep = np.where(keep > 0)
    bb = bb[keep]
    # keep的bb的置信度
    scores = scores[keep]
    cls_index = cls_index[keep]

    return bb, scores, cls_index


def create_grid(inputsize, stride, anchor_size):
    # 输入图像是正方形，宽高相同
    x = y = inputsize // stride  # 计算x和y轴数量

    '''
    https://zhuanlan.zhihu.com/p/33579211
    x = np.array([1,2,3]) #X_{x} = 3
    y = np.array([4,5,6,7]) #X_{y} = 4
    
    xv,yv = np.meshgrid( x , y )
    
    print(xv)
    print(yv)
    
    [[1 2 3]
    [1 2 3]
    [1 2 3]
    [1 2 3]]
    
    [[4 4 4]
    [5 5 5]
    [6 6 6]
    [7 7 7]]
    
    x：表示我们的一维向量(1,2,3)，他的N = 3
    y：表示我们的一维向量(4,5,6,7)，他的N = 4
    
    xv：表示x坐标轴上的坐标矩阵
    yv：表示y坐标轴上的坐标矩阵
    '''

    grid_y, grid_x = np.meshgrid(np.arange(y), np.arange(x))  # 创建 网格 grid_y是网格中每个点的y坐标，gridx是x坐标
    grid_xy = np.stack([grid_x, grid_y], axis=-1).astype(np.float)  # stack 在一起 比如两个5X5 变成5X10
    grid_xy = grid_xy.reshape((1, x * y, 1, 2))  # 然后resize 2 代表网格上一点的 x 和 y
    grid_xy_torch = torch.from_numpy(grid_xy).cuda()

    anchor_wh = anchor_size.repeat(x * y, 1, 1).unsqueeze(0).cuda()  # 网格上每个点都要放置锚点
    return grid_xy_torch, anchor_wh


# 它和v2不同在于  stride此时是一个数组[8, 16, 32]
def create_gridv3(input_size, strides, num_anchors, anchor_size):
    total_grid_xy = []  # 总计网格xy
    total_stride = []  # 总stride
    total_anchor_wh = []  # 总锚点wh
    w = h = input_size

    for idx, s in enumerate(strides):
        x, y = w // s, h // s
        grid_y, grid_x = np.meshgrid(np.arange(y), np.arange(x))
        grid_xy = np.stack([grid_x, grid_y], axis=-1).astype(np.float)
        grid_xy = grid_xy.reshape((1, x * y, 1, 2))
        # 每一个网格的左上角xy坐标
        if conf.cuda:
            grid_xy_torch = torch.from_numpy(grid_xy).cuda()
        else:
            grid_xy_torch = torch.from_numpy(grid_xy)

        # xy是根据stride划分图像后得到的横向x格数和纵向y格数，每个格子有num_anchors个anchor，2记录每一个网格的xy方向上stride
        stride = torch.ones([1, x * y, num_anchors, 2]) * s
        # 每一个网格的anchor size, 因为每一个网格都有一套anchor
        anchor = anchor_size[idx].repeat(x * y, 1, 1)

        total_grid_xy.append(grid_xy_torch)
        total_stride.append(stride)
        total_anchor_wh.append(anchor)

    if conf.cuda:
        total_grid_xy = torch.cat(total_grid_xy, 1).cuda()
        total_stride = torch.cat(total_stride, 1).cuda()
        total_anchor_wh = torch.cat(total_anchor_wh, 0).cuda()
    else:
        total_grid_xy = torch.cat(total_grid_xy, 1)
        total_stride = torch.cat(total_stride, 1)
        total_anchor_wh = torch.cat(total_anchor_wh, 0)
    total_anchor_wh = total_anchor_wh.unsqueeze(0)

    return total_grid_xy, total_stride, total_anchor_wh


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


'''

'''


def generate_anchor(input_size, stride, anchor_scale, anchor_aspect):
    return


def compute_iou(anchor_boxes, gt_box):
    '''
    计算iou
    :param anchor_boxes:ndarray -> [[c_x_s, c_y_s, anchor_w, anchor_h], ..., [c_x_s, c_y_s, anchor_w, anchor_h]].
    :param gt_box:ndarray -> [c_x_s, c_y_s, anchor_w, anchor_h].
    :return:ndarray -> [iou_1, iou_2, ..., iou_m], m is equal to the number of anchor boxes.
    '''

    # 创建一个var保存xmin ymin xmax ymax 处理anchor box
    anc_gw = anchor_boxes[:, 2]  # grid下得w
    anc_gcx = anchor_boxes[:, 0]  # grid 下的 center x
    anc_gcy = anchor_boxes[:, 1]
    anc_gh = anchor_boxes[:, 3]
    x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    # 通过中心坐标和wh算出点坐标
    x1y1_x2y2[:, 0] = anc_gcx - anc_gw / 2  # xmin
    x1y1_x2y2[:, 1] = anc_gcy - anc_gh / 2  # ymin
    x1y1_x2y2[:, 2] = anc_gcx + anc_gw / 2  # xmax
    x1y1_x2y2[:, 3] = anc_gcy + anc_gh / 2  # ymax

    # 处理gt box
    # 将box复制len(anchor_boxes)，即每个gt_box和所有anchor_boxes算iou
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)
    xy_xy = np.zeros([len(anchor_boxes), 4])
    gt_gcx = gt_box_expand[:, 0]  # grid下得w
    gt_gcy = gt_box_expand[:, 1]  # grid 下的 center x
    gt_gw = gt_box_expand[:, 2]
    gt_gh = gt_box_expand[:, 3]
    # print(xy_xy[:, 0], gt_gcx, gt_gw)  [0.] [0. 0. 0. 0. 0.] [2. 2. 2. 2. 2.]
    xy_xy[:, 0] = gt_gcx - gt_gw / 2
    xy_xy[:, 1] = gt_gcy - gt_gh / 2
    xy_xy[:, 2] = gt_gcx + gt_gw / 2
    xy_xy[:, 3] = gt_gcy + gt_gh / 2

    # 开始计算iou
    Sgt = gt_gw * gt_gh  # 面积
    Sanc = anc_gw * anc_gh
    # 重叠区域面积
    I_w = np.minimum(xy_xy[:, 2], x1y1_x2y2[:, 2]) - np.maximum(xy_xy[:, 0], x1y1_x2y2[:, 0])
    I_h = np.minimum(xy_xy[:, 3], x1y1_x2y2[:, 3]) - np.maximum(xy_xy[:, 1], x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = Sgt + Sanc - S_I + 1e-20
    IoU = S_I / U  # 得到iou, 数量等于anc box

    return IoU


def set_anchors(anchor_size):
    # 根据给定的anchor_size 创建anchors
    anchor_num = len(anchor_size)  # 5 个 anchor
    anchor_box = np.zeros([anchor_num, 4])  # 4 个特征
    for i, size in enumerate(anchor_size):
        w, h = size
        # xyhw
        anchor_box[i] = np.array([0, 0, w, h])
    return anchor_box


# 根据ground truth的label 提供的xmin, ymin, xmax, ymax 计算tx, ty, tw, th
def generate_txtytwth(gt_label, w, h, s, anchor_sizes):
    # ground truth box的坐标 这个坐标是归一化后的
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # print(xmin, ymin, xmax, ymax, w, h, s) 可知gt_label的坐标是归一化过得
    # 0.06243602931499481 0.03934426233172417 0.4790174067020416 0.3267759680747986 64 64 32  其中64是图片宽高，还原坐标到原大小图像中

    # 找到ground truth的中心并且反归一化
    centerx = (xmax + xmin) / 2 * w
    centery = (ymin + ymax) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1 or box_h < 1:
        #        print('the size of box is abnormal!')
        return False

    # 将center, width and height投射到特征图size, stride是grid的大小
    cxg = centerx / s
    cyg = centery / s
    grid_x = int(cxg)  # 转化为grid坐标  int是向下取整
    grid_y = int(cyg)
    grid_bw = box_w / s  # 在grid坐标系中box的大小
    grid_bh = box_h / s

    # 生成一个锚点box
    anchor_boxes = set_anchors(anchor_size=anchor_sizes)
    ground_truth_box = np.array([[0, 0, grid_bw, grid_bh]])

    # 计算iou， 通过anchor_boxes 以及 ground_truth_box
    iou = compute_iou(anchor_boxes, ground_truth_box)
    # We consider those anchor boxes whose IoU is more than ignore thresh,
    iou_mask = (iou > conf.ignore_threshold)

    result = []
    if iou_mask.sum() == 0:
        index = np.argmax(iou)  # 找到iou中最大的那个的
        pw, ph = anchor_sizes[index]  # 拿出iou中最大的index，取出w和h
        tx = cxg - grid_x
        ty = cyg - grid_y
        tw = np.log(grid_bw / pw)
        th = np.log(grid_bh / ph)

        weight = 2.0 - (box_w / w) * (box_h / h)  # 2.0减去(xmax - xmin)*(ymax - ymin)

        result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])

    else:
        index = np.argmax(iou)
        for idx, iou_m in enumerate(iou_mask):
            if iou_m:
                if idx == index:
                    p_w, p_h = anchor_sizes[idx]
                    tx = cxg - grid_x
                    ty = cyg - grid_y
                    tw = np.log(grid_bw / p_w)
                    th = np.log(grid_bh / p_h)
                    weight = 2.0 - (box_w / w) * (box_h / h)

                    result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])
                else:
                    # we ignore other anchor boxes even if their iou scores all higher than ignore thresh
                    result.append([index, grid_x, grid_y, 0., 0., 0., 0., -1.0, 0., 0., 0., 0.])

    return result


def gt_creator(input_size, stride, label_lists, anchor_size):
    '''
    anchor box就是从训练集中真实框（ground truth）中统计或聚类得到的几个不同尺寸的框。
    避免模型在训练的时候盲目的找，有助于模型快速收敛。假设每个网格对应k个anchor，也就是模型在训练的时候，
    它只是会在每一个网格附近找出这k种形状，不会找其他的。

    :param input_size: 图像大小
    :param stride: 模型中定义的grid大小
    :param label_lists: 一个array[] 类型 的数组，里面装的全是list， list中包括了box的信息
    :param anchor_size: 预测框的初始宽高，第一个是w，第二个是h，总数量是num*2，通过Kmeans得到
    :return:gt_tensor : ndarray -> shape = [batch_size, anchor_number, 1+1+4, grid_cell number ]
    '''
    assert len(label_lists) > 0 and len(label_lists) > 0
    batch_size = len(label_lists)
    h = w = input_size

    wgrid = w // stride  # width上以stride为步长划分了多少个网格
    hgrid = h // stride
    anchor_num = len(anchor_size)  # 这里是5
    # gt_tensor 是 有bs张图像，他们每张划分为hgrid、wgrid个网格，每个网格有anchor_num个锚点
    # 1+1+4+1+4 是指 下文gt_tensor赋值时有写他们具体是什么
    gt_tensor = np.zeros([batch_size, hgrid, wgrid, anchor_num, 1 + 1 + 4 + 1 + 4])

    for index in range(batch_size):
        for label in label_lists[index]:
            cls = int(label[-1])
            # 产生
            result = generate_txtytwth(label, w, h, stride, anchor_size)
            if result:
                for res in result:
                    # idx 索引，grid_x、grid_y坐标 tx, ty, tw, th,相对偏移 xmin, ymin, xmax, ymax box的左右下角，
                    # weight是在 generate_txtytwth 中计算出来的
                    idx, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax = res
                    # (4, 2, 2, 5, 11) 当图像大小是64
                    #                    print(gt_tensor.shape)
                    if weight > 0:
                        # 没有超出边界
                        if grid_y < gt_tensor.shape[1] and grid_x < gt_tensor.shape[2]:
                            gt_tensor[index, grid_y, grid_x, idx, 0] = 1.0  # 置信度
                            gt_tensor[index, grid_y, grid_x, idx, 1] = cls  # 分类
                            gt_tensor[index, grid_y, grid_x, idx, 2:6] = np.array([tx, ty, tw, th])  # 转换后相对于grid的坐标
                            gt_tensor[index, grid_y, grid_x, idx, 6] = weight  # generate_txtytwth的weight
                            gt_tensor[index, grid_y, grid_x, idx, 7:] = np.array([xmin, ymin, xmax, ymax])  # 左上右下坐标
                    else:
                        gt_tensor[index, grid_y, grid_x, idx, 0] = -1.
                        gt_tensor[index, grid_y, grid_x, idx, 6] = -1.

    gt_tensor = gt_tensor.reshape((batch_size, hgrid * wgrid * anchor_num, 1 + 1 + 4 + 1 + 4))

    return gt_tensor


def multi_gt_creator(input_size, strides, label_lists, anchor_size):
    '''
    多尺度scale的gt creator
    '''
    batch_size = len(label_lists)  # target的数量
    h = w = input_size  # 图像大小
    num_scale = len(strides)  # 多少个stride
    gt_tensors = []
    anchor_num = len(anchor_size) // num_scale  # 这里是5 要除以num_scale才能维度保证相等
    # print(len(anchor_size)) # 9
    for s in strides:
        gt_tensors.append(np.zeros([batch_size, h // s, w // s, anchor_num, 1 + 1 + 4 + 1 + 4]))

    # 从输入的数据中取出一个个batch，anchor size是9
    for index in range(batch_size):
        for label in label_lists[index]:
            # label 是选择index所标识的batch， 里面有1-N个不等的gtbox， label就是其中一个，
            # 格式为[0.9701110124588013, 0.51474529504776, 1.0, 0.6474530696868896, 14.0]
            cls = int(label[-1])
            xmin, ymin, xmax, ymax = label[:-1]
            centerx = (xmax + xmin) / 2 * w
            centery = (ymin + ymax) / 2 * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h

            if box_w < 1 or box_h < 1:
                continue

            anchor_boxes = set_anchors(anchor_size)  # 根据给定anchor的size来创建9个anchor
            gt_box = np.array([[0, 0, box_w, box_h]])  # gt box 落在 当前grid 中的部分， 其中0，0是指grid的中心坐标，而不是左上角
            # print('1:', anchor_boxes, '2',  gt_box)
            '''
            [[  0.     0.    32.64  47.68]
             [  0.     0.    50.24 108.16]
             [  0.     0.   126.72  96.32]
             [  0.     0.    78.4  201.92]
             [  0.     0.   178.24 178.56]
             [  0.     0.   129.6  294.72]
             [  0.     0.   331.84 194.56]
             [  0.     0.   227.84 325.76]
             [  0.     0.   365.44 358.72]]
             
              [[ 0.          0.          5.3087616  26.71209335]]
            '''
            iou = compute_iou(anchor_boxes, gt_box)  # anchor_boxes与gt_box之间的重合面积，gt中心必须落在当前grid中， iou是重合面积
            # print('iou ', iou)
            '''
            iou  [0.82888183 0.23739016 0.10568598 0.08148612 0.04053122 0.03377259
             0.01998006 0.01738006 0.00984028]
            iou  [0.14028426 0.48982266 0.32738565 0.70078055 0.33015188 0.29044428
             0.17182848 0.14946851 0.08462638]
            iou  [0.02155857 0.07527485 0.16908117 0.21929514 0.44088259 0.51205252
             0.55297463 0.8144165  0.5506742 ]
            '''
            iou_mask = (iou > conf.ignore_threshold)  # 重合超过thres 面积是1，所以重合大于0.5, 大于thres的话为1，否则视0
            # [False False False False  True False False False False]
            if iou_mask.sum() == 0:  # 如果没有iou mask
                idx = np.argmax(iou)  # 找到最大iou对应的anchor box 返回index
                #                print('utils.multi_gt_creator:', idx)
                '''
                utils.multi_gt_creator: 0
                utils.multi_gt_creator: 0
                utils.multi_gt_creator: 0
                utils.multi_gt_creator: 0
                utils.multi_gt_creator: 0
                utils.multi_gt_creator: 0
                utils.multi_gt_creator: 0
                utils.multi_gt_creator: 0
                utils.multi_gt_creator: 0
                utils.multi_gt_creator: 2
                '''
                # 计算当前iou的index，
                s_idx = idx // anchor_num  # 最大的anchor box 的idx 归一化
                anc_box_idx = idx - s_idx * anchor_num  # anchor box的index

                s = strides[s_idx]
                pw, ph = anchor_boxes[idx, 2], anchor_boxes[idx, 3]

                cxg = centerx / s
                cyg = centery / s
                grid_x = int(cxg)
                grid_y = int(cyg)

                tx = cxg - grid_x  # 相对当前grid左上角的x
                ty = cyg - grid_y
                tw = np.log(box_w / pw)
                th = np.log(box_h / ph)
                weight = 2. - (box_w / w) * (box_h / h)  # 权重

                if grid_y < gt_tensors[s_idx].shape[1] and grid_x < gt_tensors[s_idx].shape[2]:
                    gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 0] = 1.0
                    gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 1] = cls
                    gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 6] = weight
                    gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 7:] = np.array([xmin, ymin, xmax, ymax])
            else:
                best_idx = np.argmax(iou)
                # 计算
                for idx, iou_m in enumerate(iou_mask):
                    if iou_m:  # 如果有mask大于阈值，此时iou_m=True
                        if idx == best_idx:
                            s_idx = idx // anchor_num
                            anc_box_idx = idx - s_idx * anchor_num
                            s = strides[s_idx]  # 从这里可以看出，s_idx 作用是从strides 中取出正确的stride

                            pw, ph = anchor_boxes[index, 2], anchor_boxes[index, 3]
                            cxg = centerx / s
                            cyg = centery / s
                            grid_x = int(cxg)
                            grid_y = int(cyg)

                            tx = cxg - grid_x  # 相对当前grid左上角的x
                            ty = cyg - grid_y
                            tw = np.log(box_w / pw)
                            th = np.log(box_h / ph)
                            weight = 2. - (box_w / w) * (box_h / h)  # 权重

                            if grid_y < gt_tensors[s_idx].shape[1] and grid_x < gt_tensors[s_idx].shape[2]:
                                gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 0] = 1.0
                                gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 1] = cls
                                gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 2:6] = np.array([tx, ty, tw, th])
                                gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 6] = weight
                                gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 7:] = np.array(
                                    [xmin, ymin, xmax, ymax])
                        else:
                            s_idx = idx // anchor_num
                            anc_box_idx = idx - s_idx * anchor_num
                            s = strides[s_idx]  # 从这里可以看出，s_idx 作用是从strides 中取出正确的stride

                            pw, ph = anchor_boxes[index, 2], anchor_boxes[index, 3]
                            cxg = centerx / s
                            cyg = centery / s
                            grid_x = int(cxg)
                            grid_y = int(cyg)

                            gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 0] = 1.0
                            gt_tensors[s_idx][index, grid_y, grid_x, anc_box_idx, 6] = -1.

    gt_tensors = [gt.reshape((batch_size, -1, 1 + 1 + 4 + 1 + 4)) for gt in gt_tensors]
    gt_tensors = np.concatenate(gt_tensors, 1)  # 以第一维度进行concat

    return gt_tensors


def iou_score(bboxes_a, bboxes_b):
    """
        bbox_a : [B*N, 4] = [x1, y1, x2, y2]
        bbox_b : [B*N, 4] = [x1, y1, x2, y2]
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])  # 返回[x1_max, y1_max]，x1和y1不必属于同一个box
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])  # 返回[x2_min, y2_min]
    area_a = torch.prod(bboxes_a[:, :2] - bboxes_a[:, 2:], 1)  # box a 的面积
    area_b = torch.prod(bboxes_b[:, :2] - bboxes_b[:, 2:], 1)  # box b 的面积

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # 重叠部分面积
    return area_i / (area_a + area_b - area_i)  # 重叠面积除以未重叠面积
