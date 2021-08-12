# -*- coding: utf-8 -*-
# 这个训练成功，正常运作！！！ 使用的是v5v2版本进行训练
# 我们以这个模型为基础结构进行改进
import numpy as np
from tqdm import tqdm
from config import *
from models.v5v2 import YOLOv5 as net
from data.transformers import SSDAugmentation, ColorAugmentation
from utils.utils import detection_collate, multi_gt_creator
import torch.utils.data
import torch.optim as optim
from data.dataset import VOCDataset
import torch
import random
import torch.nn.functional as F
import sys
sys.path.insert(0, './yolov5')


from utils.visual_test_utils import save_box_img

conf = configv5()

dataset = VOCDataset(root=conf.voc_root,
                     transform=SSDAugmentation(size=conf.train_img_size),
                     base_transform=ColorAugmentation(size=conf.train_img_size),
                     mosaic=conf.mosaic,
                     image_sets=[('2012', 'train')])

dataloader = torch.utils.data.DataLoader(dataset, conf.batch_size, shuffle=True, collate_fn=detection_collate,
                                         pin_memory=True)
datalen = len(dataloader)
dataloader = tqdm(enumerate(dataloader))
dataiter = iter(dataloader)

model = net(input_size=conf.train_img_size, nc=conf.class_num, anchors=conf.anchor_size_voc)
if conf.cuda:
    model = model.cuda().train()
else:
    model = model.train()

if conf.pretrained:  # 加载预训练模型
    pretrained_file = torch.load(conf.pretrained_path)
    print(pretrained_file.keys())
    model_dict = pretrained_file['model'].float.state_dict()

exit(0)
init_lr = temp_lr = conf.lr

optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(.5, .999))

epoch_size = len(dataset) // conf.batch_size
# lf = lambda x: (1 - x / (conf.max_epoch - 1)) * (1.0 - 0.2) + 0.2
# schedular = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, conf.max_epoch * datalen)

for epoch in range(conf.start_epoch, conf.max_epoch):
    for it, (images, targets) in dataloader:
        if conf.cuda:
            images = images.cuda()


        # 加入多scale trick  为了增强模型的鲁棒性，YOLOv2采用了多尺度输入训练策略，具体来说就是在训练过程中每间隔一定的iterations之后改变模型的输入图片大小。
        # 每间隔一定的iterations之后改变模型的输入图片大小 只需要修改对最后检测层的处理就可以重新训练。
        # 由于YOLOv2的下采样总步长为32，输入图片大小选择一系列为32倍数的值：输入图片最小为 320
        # 此时对应的特征图大小为 10X10， 输入图片最大为608X608， 对应的特征图大小为19X19
        # 采用Multi-Scale Training策略，YOLOv2可以适应不同大小的图片，并且预测出很好的结果。
        if conf.multi_scale and (it + 1) % 10 == 0:
            r = conf.random_size_range
            conf.train_img_size = random.randint(r[0], r[1]) * 32
            model.set_grid(conf.train_img_size)
        if conf.multi_scale:
            images = F.upsample(images, size=conf.train_img_size, mode='bilinear', align_corners=False)

        targets = [t.tolist() for t in targets]  # 将target转成指定形式

        # 在v2版本，使用的是单一gtbox

        targets = multi_gt_creator(conf.train_img_size, model.stride, targets, conf.anchor_size_voc)

        targets = torch.Tensor(targets).float().cuda() if conf.cuda else torch.Tensor(targets).float()

        ys, ym, yl = model(images)

        conf_loss, cls_loss, bbox_loss, iou_loss = model.computeLoss(preds=[ys, ym, yl], target=targets)
        # 对loss 进行 scale
        total_loss = conf_loss/conf.scale_factor + cls_loss + bbox_loss/conf.scale_factor + iou_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        schedular.step()

        print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
              '[Loss: obj %.2f || cls %.2f || bbox %.2f || iou %.2f || total %.2f || size %d]'
              % (epoch + 1, conf.max_epoch, it, epoch_size, optimizer.param_groups[0]['lr'],
                 conf_loss.item(),
                 cls_loss.item(),
                 bbox_loss.item(),
                 iou_loss.item(),
                 total_loss.item(),
                 conf.train_img_size),
              flush=True)

    if (epoch + 1) % 2 == 0:
        print(f'保存结果{epoch} : ')
        images, targets = next(dataiter)
        ind = random.randint(0, conf.batch_size - 1)
        image = images[ind].unsqueeze(0)
        target = targets[ind]
        bb, score, cls = model(image.cuda(), eva=True)
        # 画出conf最大的10个box
        sort_top = score.argsort()[-10:]
        bb_top = bb[sort_top]  # （10， 4）
        cls_top = cls[sort_top].reshape((-1, 1))  # (10,)

        pred = np.hstack((bb_top, cls_top))

        save_box_img(image, target, conf.train_img_size,
                     index2=conf.version + '_' + str(epoch + 1) + '_' + str(ind) + '_gt')
        save_box_img(image, pred, conf.train_img_size,
                     index2=conf.version + '_' + str(epoch + 1) + '_' + str(ind) + '_pred')

        torch.save(model.state_dict(), f'result/model/voc_yolov5/model_{epoch + 1}.pt')
        torch.save(optimizer.state_dict(), f'result/model/voc_yolov5/optim_{epoch + 1}.pt')
