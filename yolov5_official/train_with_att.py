import math
from copy import deepcopy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import yolov5_official.config as cfg
import yaml, cv2
import yolov5_official.val as val
from torch.cuda import amp
from yolov5_official.EMAModel import ModelEMA
from yolov5_official.datasets import create_dataloader
from yolov5_official.utils import one_cycle, ComputeLoss, labels_to_class_weights, fitness, check_img_size, check_anchors, load_pt
from yolov5_official.model import Model, ModelWithAtten
import os

RANK = int(os.getenv('RANK', -1))  # -1

if not os.path.exists(cfg.save_dir + cfg.name):
    os.mkdir(cfg.save_dir + cfg.name)


# 读取超参数和数据参数  
with open(cfg.hyp) as f:
    hyperparams = yaml.safe_load(f)
with open(cfg.data) as f:
    dataparams = yaml.safe_load(f)
cfg.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

# 把此次运行的参数都保存下来
with open(cfg.save_dir + cfg.name + 'hyp.yaml', 'w') as f:
    yaml.safe_dump(hyperparams, f)
with open(cfg.save_dir + cfg.name + 'cfgig.yaml', 'w') as f:
    yaml.safe_dump(cfg.__name__, f)

# 创建模型
model = ModelWithAtten(cfg.cfg, ch=3, nc=dataparams.get('nc'), anchors=hyperparams.get('anchors', None))

model.gr = 1.0
model.train()
train_path = dataparams.get('train')  # 训练路径
test_path = dataparams.get('val')  # 验证路径

# optimizer 的参数
nbs = 64  # nominal batch size
accumulate = max(round(nbs / cfg.batch_size), 1)  # accumulate loss before optimizing
hyperparams['weight_decay'] *= cfg.batch_size * accumulate / nbs  # scale weight_decay

# 冻结参数
freeze = []
for k, v in model.named_parameters():
    v.requires_grad = True  # train all layers
    if any(x in k for x in freeze):
        print('freezing %s' % k)
        v.requires_grad = False

# 根据模块来选择不同的optimizer
pg0, pg1, pg2 = [], [], []  # 优化函数的参数
for k, v in model.named_modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)
    if isinstance(v, nn.BatchNorm2d):
        pg0.append(v.weight)
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)

# 对应不同module的参数使用不同的optimizer
if cfg.adam:
    optimizer = optim.Adam(pg0, lr=hyperparams['lr0'],
                           betas=(hyperparams['momentum'], 0.999))  # 调整beta1
else:
    optimizer = optim.SGD(pg0, lr=hyperparams['lr0'], momentum=hyperparams['momentum'], nesterov=True)

optimizer.add_param_group({'params': pg1, 'weight_decay': hyperparams['weight_decay']})
optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
del pg0, pg1, pg2

# 制定 lr 的更新策略
if cfg.linear_lr:
    lf = lambda x: (1 - x / (cfg.epochs - 1)) * (1.0 - hyperparams['lrf']) + hyperparams['lrf']  # linear
else:
    lf = one_cycle(1, hyperparams['lrf'], cfg.epochs)

scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

start_epoch = 0

if cfg.pretrained:
    load_pt(model, cfg.weights)
    print('loaded!')

# Image sizes
gs = max(int(model.stride.max()), 32)  # grid size (max stride)
nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])

imgsz = check_img_size(cfg.img_size, gs, gs * 2)

# 读取数据
dataloader, dataset = create_dataloader(train_path, cfg.img_size, cfg.batch_size // cfg.world_size, gs, cfg,
                                        hyp=hyperparams, augment=True, cache=cfg.cache_images, rect=cfg.rect,
                                        world_size=cfg.world_size, workers=cfg.workers,
                                        image_weights=cfg.image_weights, quad=cfg.quad, prefix='train: ')
mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # 最大标签类别
nb = len(dataloader)  # batch数量
dataiter = iter(dataloader)

if RANK in [-1, 0]:
    val_loader = create_dataloader(test_path, cfg.img_size, cfg.batch_size // cfg.world_size * 2, gs, cfg,
                                   hyp=hyperparams, cache=cfg.cache_images, rect=True,
                                   world_size=cfg.world_size, workers=cfg.workers, pad=.5, prefix='val: ')[0]

    # 自动计算锚框功能 从而使对目标的区域识别更加准确
    if not cfg.resume:  # 如果不是resume name自动计算锚框
        check_anchors(dataset, thr=hyperparams.get('anchor_t'), imgsz=imgsz, model=model)

# 该方法在训练网络时将单精度（FP32）与半精度(FP16)结合在一起，并使用相同的超参数实现了与FP32几乎相同的精度。
# 只有支持tensor core的CUDA硬件才能使用
amp_scaler = amp.GradScaler(enabled=cfg.use_gpu)

hyperparams['box'] *= 3. / nl  # scale to layers
hyperparams['cls'] *= dataparams.get('nc') / 80. * 3. / nl  # scale to classes and layers
hyperparams['obj'] *= (cfg.img_size / 640) ** 2 * 3. / nl  # scale to image size and layers
hyperparams['label_smoothing'] = cfg.label_smoothing
model.nc = dataparams.get('nc')  # 类别数
model.hyp = hyperparams  # 超参数
model.gr = 1.0  # iou损失ratio
model.class_weights = labels_to_class_weights(dataset.labels, dataparams.get('nc')) * dataparams.get(
    'nc')  # 根据标签的数量获得权重， 从而减少因为标签数量的不平均导致训练失衡
# 初始化 loss 计算函数
compute_loss = ComputeLoss(model, hyperparams)
best_fitness = 0.0
results = (0, 0, 0, 0, 0, 0, 0)  # results 中 记录了P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
nw = max(round(hyperparams['warmup_epochs'] * nb), 1000)

optimizer.zero_grad()

for e in range(start_epoch, cfg.epochs):
    model.train()
    if cfg.use_gpu:
        mloss = torch.zeros(4).cuda()  # 平均loss
    else:
        mloss = torch.zeros(4)

    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nb)
    pbar.desc = 'training ......'
    for i, (imgs, targets, paths, _) in pbar:
        # img : torch.Size([4, 3, 640, 640]) , targets : torch.Size([58, 6])
        # print(imgs.shape, targets.shape, paths)
        ni = i + nb * e
        imgs = imgs.cuda().float() / 255.0 if cfg.use_gpu else imgs.float() / 255.0

        # 加入了 warm up 策略
        # 有助于减缓模型在初始阶段对mini-batch的提前过拟合现象，保持分布的平稳
        # 有助于保持模型深层的稳定性
        if ni <= nw:
            xi = [0, nw]  # x interp
            # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            accumulate = max(1, np.interp(ni, xi, [1, nbs / cfg.batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [hyperparams['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(e)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [hyperparams['warmup_momentum'], hyperparams['momentum']])

        if cfg.multi_scale:
            sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        if cfg.use_gpu:
            targets = targets.cuda()
        pred = model(imgs)  # forward
        loss, loss_items = compute_loss(pred, targets)  # loss scaled by batch_size
        if cfg.quad:
            loss *= 4.

        amp_scaler.scale(loss).backward()
        if ni % accumulate == 0:
            amp_scaler.step(optimizer)  # 就等于 optimizer.step()
            amp_scaler.update()
            optimizer.zero_grad()

        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

        # 打印损失 4个损失等  target 数量 和image的大小
        s = ('%10s' + '%10.4g' * 6) % (f'{e + 1}/{cfg.epochs - 1}', *mloss, targets.shape[0], imgs.shape[-1])
        pbar.desc = s
        # end batch

    scheduler.step()  # 更新学习率


    # 每运行2个epoch， 则自动保存模型， 并且打印测试结果
    if (e + 1) % cfg.eval_interval == 0:  # 经过2个epoch之后保存模型
        ckpt = {'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }

        # 保存当前结果的训练数， 模型state和optimizer的state
        torch.save(ckpt, cfg.save_dir + cfg.name + f'v5s_offical_model_{e + 1}.pt')
        del ckpt
    # end epoch
