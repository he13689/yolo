import os
import random
import numpy as np

import torch.utils.data
import torch.optim as optim
from config import *
import models.model_config as cf
from data.transformers import SSDAugmentation, ColorAugmentation, BaseTransform
from data.dataset import VOCDataset
from data.vocapi_evaluator import VOCAPIEvaluator
from utils.utils import detection_collate, gt_creator, multi_gt_creator
import torch.nn.functional as F
from utils.visual_test_utils import save_box_img

if conf.version == 'yolov2_d19':  # checked
    from models.v2d19 import YOLOv2 as net

    cfg = cf.yolov2_d19_cfg

elif conf.version == 'yolov2_r50':  # checked
    from models.v2r50 import YOLOv2 as net

    cfg = cf.yolov2_r50_cfg

elif conf.version == 'yolov2_slim':  # checked
    from models.v2slim import YOLOv2 as net

    cfg = cf.yolov2_slim_cfg

elif conf.version == 'yolov3':  # checked
    from models.v3 import YOLOv3 as net

    cfg = cf.yolov3_d53_cfg

elif conf.version == 'yolov3_spp':
    from models.v3spp import YOLOv3 as net

    cfg = cf.yolov3_d53_cfg

elif conf.version == 'yolov4':
    from models.v4 import YOLOv4 as net

    cfg = cf.yolov4_cfg

elif conf.version == 'yolov3_tiny':
    from models.v3tiny import YOLOv3 as net

    cfg = cf.yolov3_tiny_cfg

elif conf.version == 'yolov5':
    from models.v3tiny import YOLOv3 as net

    cfg = cf.yolov5_cfg

else:
    print('Unknown model name...')
    exit(0)

conf.train_img_size = cfg.get('train_size')
conf.val_img_size = cfg.get('val_size')

path_to_save = os.path.join(conf.save_folder, conf.dataset + '_' + conf.version)
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

if conf.dataset == 'voc':
    # 暂时没有, ('2012', 'trainval')
    dataset = VOCDataset(root=conf.voc_root,
                         transform=SSDAugmentation(size=conf.train_img_size),
                         base_transform=ColorAugmentation(size=conf.train_img_size),
                         mosaic=conf.mosaic,
                         image_sets=[('2007', 'trainval')])  # 是否使用mosaic增强方法

    evaluator = VOCAPIEvaluator(data_root=conf.test_root,
                                img_size=cfg['val_size'],
                                transform=BaseTransform(cfg['val_size']),
                                labelmap=conf.classes)
elif conf.dataset == 'coco':
    print('unknow dataset !! Only support voc  !!')
    exit(0)
else:
    print('unknow dataset !! Only support voc  !!')
    exit(0)

print('Training model on:', conf.version)
print('The dataset size:', len(dataset))
print("----------------------------------------------------------")

dataloader = torch.utils.data.DataLoader(dataset, conf.batch_size, shuffle=True, collate_fn=detection_collate,
                                         pin_memory=True)
dataiter = iter(dataloader)
datalen = len(dataloader)
conf.anchor_size = cfg['anchor_size_voc']
# pretrain = False，不使用预训练好的weight， trainable表示网络是可以被训练的
model = net(input_size=conf.train_img_size, num_classes=conf.class_num, trainable=True, anchor_size=conf.anchor_size)
model = model.cuda().train()

if conf.resume is not None:
    model.load_state_dict(torch.load(conf.resume))

init_lr = temp_lr = conf.lr
optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(.937, .999))
epoch_size = len(dataset) // conf.batch_size
schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, datalen * 200)

for epoch in range(conf.start_epoch, conf.max_epoch):
    # 不同阶段使用不同大小的lr
    if epoch in cfg.get('lr_epoch') and not conf.no_warm_up:
        # 将学习率变为10分之一
        temp_lr = temp_lr * 0.1
        for params in optimizer.param_groups:
            params['lr'] = temp_lr

    for it, (images, targets) in enumerate(dataloader):
        # 对学习率采取warm up策略
        if not conf.no_warm_up:
            if epoch < conf.wp_epoch:
                # 根据当前epoch动态计算学习率
                temp_lr = init_lr * pow((it + epoch * epoch_size) * 1. / (conf.wp_epoch * epoch_size), 4)
                for params in optimizer.param_groups:
                    params['lr'] = temp_lr
            elif epoch == conf.wp_epoch and it == 0:
                # 到达设定wp_epoch后
                temp_lr = init_lr
                for p in optimizer.param_groups:
                    p['lr'] = temp_lr

        images = images.cuda()

        # 加入多scale trick  为了增强模型的鲁棒性，YOLOv2采用了多尺度输入训练策略，具体来说就是在训练过程中每间隔一定的iterations之后改变模型的输入图片大小。
        # 每间隔一定的iterations之后改变模型的输入图片大小 只需要修改对最后检测层的处理就可以重新训练。
        # 由于YOLOv2的下采样总步长为32，输入图片大小选择一系列为32倍数的值：输入图片最小为 320
        # 此时对应的特征图大小为 10X10， 输入图片最大为608X608， 对应的特征图大小为19X19
        # 采用Multi-Scale Training策略，YOLOv2可以适应不同大小的图片，并且预测出很好的结果。
        if conf.multi_scale and (it + 1) % 10 == 0:
            r = cfg['random_size_range']
            conf.train_img_size = random.randint(r[0], r[1]) * 32
            model.set_grid(conf.train_img_size)
        if conf.multi_scale:
            images = F.upsample(images, size=conf.train_img_size, mode='bilinear', align_corners=False)

        targets = [t.tolist() for t in targets]  # 将target转成指定形式

        # 在v2版本，使用的是单一gtbox
        if conf.version == 'yolov2_d19' or conf.version == 'yolov2_r50' or conf.version == 'yolov2_slim':
            targets = gt_creator(input_size=conf.train_img_size, stride=model.stride, label_lists=targets,
                                 anchor_size=conf.anchor_size)  # 创建gt box
        else:
            # 多gt
            targets = multi_gt_creator(conf.train_img_size, model.stride, targets, conf.anchor_size)

        targets = torch.Tensor(targets).float().cuda()

        conf_loss, cls_loss, bbox_loss, iou_loss = model(images, targets)
        total_loss = conf_loss + cls_loss + bbox_loss + iou_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if conf.no_warm_up:
            schedular.step()

        if it % 10 == 0:
            print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                  '[Loss: obj %.5f || cls %.5f || bbox %.5f || iou %.5f || total %.5f || size %d ]'
                  % (epoch + 1, conf.max_epoch, it, datalen,
                     optimizer.param_groups[0]['lr'] if conf.no_warm_up else temp_lr,
                     conf_loss.item(),
                     cls_loss.item(),
                     bbox_loss.item(),
                     iou_loss.item(),
                     total_loss.item(),
                     conf.train_img_size),
                  flush=True)

    # 每过conf.eval_epoch
    if (epoch + 1) % conf.eval_epoch == 0:
        print('暂无eval，改为visualize')
        # torch.Size([32, 3, 32, 32]),  32 个5X5 应该是5个box，每个xywh加conf
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

    if (epoch + 1) % (conf.eval_epoch * 2) == 0:
        torch.save(model.state_dict(), f'result/model/voc_yolov2_r50/model_{epoch + 1}.pt')
        torch.save(optimizer.state_dict(), f'result/model/voc_yolov2_r50/optim_{epoch + 1}.pt')
