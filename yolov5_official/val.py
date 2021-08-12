"""
Validate a trained YOLOv5 model accuracy on a custom dataset
Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolov5_official.utils import coco80_to_coco91_class, non_max_suppression, scale_coords, xywh2xyxy, process_batch, \
    save_one_txt, save_one_json, ap_per_class
# from yolov5_official.utils import diou_non_max_suppression as non_max_suppression
from yolov5_official.utils import diou_non_max_suppression

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path


# 计算模型的各项指标 以 判定模型的好坏
@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        save_txt=True,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        compute_loss=None,
        ):
    
    half = half and device != 'cpu'
    
    # 开启半精度。直接可以加快运行速度、减少GPU占用，并且只有不明显的accuracy损失
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = False  # 是否使用的是coco数据集
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()  # iouv中元素数量

    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # 80 class标签转 91class
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 结果 数据
    p, r, f1, mp, mr, map50, map95, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(4, device=device)  # 损失的值
    jdict, stats, ap, ap_class = [], [], [], []  # 存储计算所需要的中间数据
    # 读取数据  从val loader中
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='validating please wait')):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape

        # 运行 得到输出结果  augment设为false  输出是
        # train_out : 三套对应不同大小特征图的先验框预测结果
        # out : 经inplace等操作后的输出结果 [16, 25200, 85]
        out, train_out = model(img, augment=augment)

        # 计算 loss
        if compute_loss:
            # 使用loss 这个 tensor 记录 三个loss和总loss
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, total 他们都是detach的

        # 使用NMS过滤框
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # 转为像素形式
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        # 每幅图片中筛选出300个nms框 [300 6]
        # 返回格式 16长度list， 每个里面都是[300 6]   即对于每张图片，我们都通过nms筛选出300个框
        nms_out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        nms_out = diou_non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        # 每个图片的数据  看看输出的框  out中存储的是所有先验框的类别结果
        for si, pred in enumerate(nms_out):
            # 取出targets[:, 0] 也就是 target中每个图片(对应16个index)所对应的目标 可以是一个图片样本中包括多个target 形状是[72,6]
            labels = targets[targets[:, 0] == si, 1:]  # 72个target中属于图片si的所有target  取出的是conf以及位置
            nl = len(labels)  # label 数量
            tcls = labels[:, 0].tolist() if nl else []  # label所属的类别， 是哪一类
            path, shape = Path(paths[si]), shapes[si][0]  # 这张图片所在路径  这张图片的大小
            seen += 1  # 所见过的图片+1

            #  如果没有预测结果
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # 如果是单一 类别
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # 原生空间预测  改变了predn的内部数据
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            # 评价
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes的坐标位置
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # 原生空间标签
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # 原生空间标签
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # 预测正确的框 置信度 预测类别 真实类别
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
            # 记录结果
            if save_txt:
                s = f"\n labels has saved to {save_dir + 'labels'}" if save_txt else ''
                print(s)
                # 问题在于 predn 中的位置信息始终为nan
                save_one_txt(predn, save_conf, shape, file=save_dir + 'labels/' + (path.stem + '.txt'))
                
            if save_json:
                # 将实验结果全部保存在jdict中
                save_one_json(predn, jdict, path, class_map)
    

    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 将stats的结果 concat在一起
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map95 = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # 每个class 有多少个 targets 显示方式是 [cls0 的数量， cls1 的数量， ... ，cls13 的数量]  这个nt向量最少也有nc的长度
    else:
        nt = torch.zeros(1)

    # 打印结果标题
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    print(s)
    # 按格式打印 结果数据
    pf = '%20s' + '%11f' * 2 + '%11.3g' * 4
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map95))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image

    # 保存json文件
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir + f"{w}_predictions.json")  # predictions json
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
            try:
                anno = COCO(anno_json)  # init annotations api
                pred = anno.loadRes(pred_json)  # init predictions api
                eval = COCOeval(anno, pred, 'bbox')
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in
                                          dataloader.dataset.img_files]  # image IDs to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                map95, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            except Exception as e:
                print(f'pycocotools unable to run: {e}')


    # 返回结果
    model.float()
    maps = np.zeros(nc) + map95

    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map95, *(loss.cpu() / len(dataloader)).tolist()), maps, t
