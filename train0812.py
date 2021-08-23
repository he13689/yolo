'''
稳定版 的 0.1.1 version ，请不要随意改动   我们的algae数据集使用了加强 数据集样本数量为2倍
0812进行更新， 更新内容亟待调试

更改明细：
启用 Diou nms 和CIoU Loss
在Algae上 13 epoch 出现 val结果
改变batch size 为 64  （4个16之后更新）
使用finetune参数 其中lr0等关键参数都经过了调试，能够更好地训练模型
改变了模型加载参数的方式和位置
更新了损失函数的计算方式， 加入了正样本的权重 加入了自动平衡
降低了val中的threshold 0.7 to 0.6
使用scratch 参数从新训练 hyp.scratch.yaml
改变存储方式 非只保存
打开label smoothing
打开multi scale

8.12 更新  训练方法更新
训练计划 对比v5s和我们改造过的v5s之间性能差距并进行分析
                对比v5s 和 带transformer的v5s之间的性能对比
                使用v5m 进行训练，其中缩小bs以便能够训练

datasets 中
531 略微更改了 mix up 的计算方法
557 augment_hsv 效果并不明显 所以去掉加快训练速度

8.13 更新
VAL 模块更新
改用 新的超参数文件 scratch_new
检测 mosaic 是否正确启动
将 固定 random seed 改为可选
将zero grad移到模型外
修复 save img box 函数中存在的错误
！！！ 将obj loss 改为 focal loss可以测试一下 有可能产生较大涨点
更新了两幅图像的标签

现在需要继续优化一下 模型的 读取保存模块，保证模型读取保存的顺利完成

8.14 加入cudnn.benchmark, cudnn.deterministic的设定


验证得知 我们的dataset module 没有问题
Val module 没有问题
NMS 没有问题
Compute loss 没有问题
不会是module的问题 否则无法加载
修改了compute loss 中将参数 .long导致模型不精确的错误
修改了label to class weight的计算方法

'''
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.backends import cudnn
from tqdm import tqdm
import yolov5_official.config as cfg
import yaml, random, math
import yolov5_official.val as val
from torch.cuda import amp
from yolov5_official.EMAModel import ModelEMA

from yolov5_official.datasets import create_dataloader

from yolov5_official.utils import one_cycle, ComputeLoss, labels_to_class_weights, fitness, check_img_size, \
    check_anchors, labels_to_image_weights, save_box_img, intersect_dicts
from yolov5_official.model import Model
import os

RANK = int(os.getenv('RANK', -1))  # -1
cfg.world_size = int(os.getenv('WORLD_SIZE', 1))  # 1

if not os.path.exists(cfg.save_dir + cfg.name):
    os.mkdir(cfg.save_dir + cfg.name)

# 读取超参数和数据参数
with open(cfg.hyp, encoding='utf-8') as f:
    hyperparams = yaml.safe_load(f)
# 把此次运行的参数都保存下来
with open(cfg.save_dir + cfg.name + 'hyp.yaml', 'w') as f:
    yaml.safe_dump(hyperparams, f)
with open(cfg.save_dir + cfg.name + 'cfgig.yaml', 'w') as f:
    yaml.safe_dump(cfg.__name__, f)

# 设定随机数 保证结果可重复
if not cfg.random_seed:
    seed = 1 + RANK
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False

with open(cfg.data, encoding='ascii', errors='ignore') as f:
    dataparams = yaml.safe_load(f)

names = ['item'] if cfg.single_cls and len(dataparams['names']) != 1 else dataparams['names']

if cfg.resume != '':
    load_ckpt = torch.load(cfg.resume, map_location=cfg.device)
    model = Model(cfg.cfg or load_ckpt['model'].yaml, ch=3, nc=hyperparams.get('nc'),
                  anchors=hyperparams.get('anchors')).to(cfg.device)
    exc = ['anchor'] if (cfg or hyperparams.get('anchors')) and not cfg.resume else []
    statedict = load_ckpt['model']
    statedict = intersect_dicts(statedict, model.state_dict(), exclude=exc)  # 加入exc
    model.load_state_dict(statedict, strict=False)  # load statedict 改動之後模型運行成功了
    print('resume model ! ')
elif cfg.pretrained:
    # 加载预训练模型  在models 里面创建了yolo和common文件以便读取迁移学习的模型yolov5s.pt等数据，因为它们是直接load模型，而不只是state，所以要在对应位置有创建模型的代码才可以load
    load_ckpt = torch.load(cfg.weights, map_location=cfg.device)
    print(load_ckpt['model'].yaml)
    model = Model(cfg.cfg or load_ckpt['model'].yaml, ch=3, nc=hyperparams.get('nc'),
                  anchors=hyperparams.get('anchors')).to(cfg.device)
    exc = ['anchor'] if (cfg or hyperparams.get('anchors')) and not cfg.resume else []
    statedict = load_ckpt['model'].float().state_dict()
    statedict = intersect_dicts(statedict, model.state_dict(), exclude=exc)  # 加入exc
    model.load_state_dict(statedict, strict=False)  # load statedict 改動之後模型運行成功了
    print('loaded model ! ')
else:
    model = Model(cfg.cfg, ch=3, nc=dataparams.get('nc'), anchors=hyperparams.get('anchors', None)).to(cfg.device)

model.train()
train_path = dataparams.get('train')  # 训练路径
test_path = dataparams.get('val')  # 验证路径

# 冻结参数
freeze = []
for k, v in model.named_parameters():
    v.requires_grad = True  # train all layers
    if any(x in k for x in freeze):
        print('freezing %s' % k)
        v.requires_grad = False

# optimizer 的参数
nbs = 64  # nominal batch size
accumulate = max(round(nbs / cfg.batch_size), 1)  # 累加loss 相当于把batch扩大了 避免内存不足导致训练失败的问题
print(f'accumulate = {accumulate}')
hyperparams['weight_decay'] *= cfg.batch_size * accumulate / nbs  # scale 结果0.0005

# 根据模块来选择不同的optimizer
pg0, pg1, pg2 = [], [], []  # 优化函数的参数
for v in model.modules():
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


# 加入EMA  增加模型的鲁棒性  提高最终模型在测试数据上的表现
# 滑动平均(exponential moving average)，或者叫做指数加权平均(exponentially weighted moving average)，
# 可以用来估计变量的局部均值，使得变量的更新与一段时间内的历史取值有关。
# 滑动平均可以看作是变量的过去一段时间取值的均值，相比对变量直接赋值而言，滑动平均得到的值在图像上更加平缓光滑，
# 抖动性更小，不会因为某次的异常取值而使得滑动平均值波动很大
ema = ModelEMA(model) if RANK in [-1, 0] else None

start_epoch = 0  # 起始点
best_fitness = 0.0  # 保存 评估后效果最好的模型

if cfg.pretrained:
    if load_ckpt['optimizer'] is not None:
        optimizer.load_state_dict(load_ckpt['optimizer'])
        best_fitness = load_ckpt['best_fitness']

        # 加载EMA模型
    if ema and load_ckpt.get('ema'):
        ema.ema.load_state_dict(load_ckpt['ema'].float().state_dict())  # 使用半精度保存的，现在要还原精度
        ema.updates = load_ckpt['updates']

    start_epoch = load_ckpt['epoch'] + 1
    # 如果已经训练完， 就不能继续训练了，可以重新设置以fine tune模型参数

    print('loaded optimizer ! ')

# 节约内存
del load_ckpt, statedict

gs = max(int(model.stride.max()), 32)  # 网格大小 grid size
nl = model.model[-1].nl  # detect中有多少个layer
imgsz = check_img_size(cfg.img_size, gs, gs * 2)

# 读取数据
dataloader, dataset = create_dataloader(train_path, imgsz, cfg.batch_size // cfg.world_size, gs, cfg,
                                        hyp=hyperparams, augment=True, cache=cfg.cache_images, rect=cfg.rect,
                                        world_size=cfg.world_size, workers=cfg.workers,
                                        image_weights=cfg.image_weights, quad=cfg.quad, prefix='train: ')
mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # 最大标签类别
nb = len(dataloader)  # batch数量

if RANK in [-1, 0]:
    val_loader = create_dataloader(test_path, imgsz, cfg.batch_size // cfg.world_size * 2, gs, cfg,
                                   hyp=hyperparams, cache=cfg.cache_images, rect=True,
                                   world_size=cfg.world_size, workers=cfg.workers, pad=.5, prefix='val: ')[0]

    # 自动计算锚框功能 从而使对目标的区域识别更加准确
    if not cfg.resume:  # 如果不是resume name自动计算锚框
        labels = np.concatenate(dataset.labels, 0)
        if not cfg.noautoanchor:
            check_anchors(dataset, thr=hyperparams.get('anchor_t'), imgsz=imgsz, model=model)
        model.half().float()

hyperparams['box'] *= 3. / nl  # scale to layers
hyperparams['cls'] *= dataparams.get('nc') / 80. * 3. / nl  # scale to classes and layers
hyperparams['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
hyperparams['label_smoothing'] = cfg.label_smoothing
model.nc = dataparams.get('nc')  # 类别数
model.hyp = hyperparams  # 超参数
model.gr = 1.0  # iou损失ratio
model.names = names
model.class_weights = labels_to_class_weights(dataset.labels, dataparams.get('nc')).to(cfg.device) * dataparams.get(
    'nc')  # 根据标签的数量获得权重， 从而减少因为标签数量的不平均导致训练失衡

last_update = -1
maps = np.zeros(dataparams.get('nc'))
results = (0, 0, 0, 0, 0, 0, 0)  # results 中 记录了P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
nw = max(round(hyperparams['warmup_epochs'] * nb), 1000)
scheduler.last_epoch = start_epoch - 1  # 不移动

# 该方法在训练网络时将单精度（FP32）与半精度(FP16)结合在一起，并使用相同的超参数实现了与FP32几乎相同的精度。
# 只有支持tensor core的CUDA硬件才能使用
amp_scaler = amp.GradScaler(enabled=cfg.use_gpu)
# 初始化 loss 计算函数
compute_loss = ComputeLoss(model, autobalance=cfg.lr_autobalance)
# optimizer.zero_grad()

for e in range(start_epoch, cfg.epochs):
    model.train()

    # 如果训练剩余小于15 epoch 那么关闭mosaic 数据增强
    if e >= cfg.epochs - 15:
        hyperparams['mosaic'] = 0.0

    # 加入image weight
    if cfg.image_weights:
        if RANK in [-1, 0]:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / dataparams.get('nc')  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=dataparams.get('nc'), class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  #

    if cfg.use_gpu:
        mloss = torch.zeros(3).cuda()  # 平均loss
    else:
        mloss = torch.zeros(3)

    pbar = enumerate(dataloader)
    if RANK in [-1, 0]:
        pbar = tqdm(pbar, total=nb)
    pbar.desc = 'training ......'
    optimizer.zero_grad()  # 部分 batch的 loss会被清除 可以考虑放在epoch之外

    for i, (imgs, targets, paths, _) in pbar:
        # img : torch.Size([4, 3, 640, 640]) , targets : torch.Size([58, 6])
        # print(imgs.shape, targets.shape, paths)

        ni = i + nb * e
        imgs = imgs.to(cfg.device, non_blocking=True).float() / 255.0

        # 加入了 warm up 策略
        # 有助于减缓模型在初始阶段对mini-batch的提前过拟合现象，保持分布的平稳
        # 有助于保持模型深层的稳定性
        if ni <= nw:
            xi = [0, nw]  # x interp
            accumulate = max(1, np.interp(ni, xi, [1, nbs / cfg.batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [hyperparams['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(e)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [hyperparams['warmup_momentum'], hyperparams['momentum']])

        # 使用multi scale image 自适应调节输入图像大小
        if cfg.multi_scale:
            # 对输入image进行缩放 倍率计算如下
            sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        if cfg.use_gpu:
            targets = targets.cuda()

        with amp.autocast(enabled=cfg.use_gpu):
            pred = model(imgs)  # forward
            # lbox, lobj, lcls
            loss, loss_items = compute_loss(pred, targets)  # loss scaled by batch_size
            if cfg.quad:
                loss *= 4.

        amp_scaler.scale(loss).backward()

        if ni - last_update >= accumulate:
            amp_scaler.step(optimizer)  # 就等于 optimizer.step()
            amp_scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
            last_update = ni

        # 实时更新
        # amp_scaler.step(optimizer)  # 就等于 optimizer.step()
        # amp_scaler.update()
        # optimizer.zero_grad()
        # if ema:
        #     ema.update(model)

        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

        # 打印损失 4个损失   target label数量  和image的大小
        s = ('%10s' + '%10.4g' * 5) % (f'{e + 1}/{cfg.epochs - 1}', *mloss, targets.shape[0], imgs.shape[-1])
        pbar.set_description(s)
        # end batch

    # lr = [x['lr'] for x in optimizer.param_groups]
    scheduler.step()  # 更新学习率  已被关闭 效果不好  可以通过将last epoch 设为其他值来读取

    # 每个epoch 都会对模型 进行评估
    if RANK in [-1, 0]:
        # 计算mAP
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])  # ema  更新属性
        final_epoch = e + 1 == cfg.epochs
        # if final_epoch:  # Calculate mAP
        results, maps, _ = val.run(dataparams,
                                   model=ema.ema,
                                   single_cls=cfg.single_cls,
                                   dataloader=val_loader,
                                   save_dir=cfg.save_dir + cfg.name,
                                   save_json=False,
                                   plots=False,
                                   # compute_loss=compute_loss,
                                   compute_loss=None,  # 不计算val loss 减少运行时间
                                   conf_thres=1e-3,
                                   iou_thres=0.7)

        # 更新并记录最好的mAP
        fi = fitness(np.array(results).reshape(1, -1), w=[0., 0., 0.2, 0.8])  # [P, R, mAP@.5, mAP@.5-.95]
        # 如果当前的fitness大于最好的fitness，就是我们当前最好的模型
        if fi > best_fitness:
            best_fitness = fi
            ckpt = {'epoch': e,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ema': deepcopy(ema.ema).half(),  # 将所有的浮点参数和缓冲转换为半浮点(half)数据类型. 节约空间
                    'updates': ema.updates,
                    'best_fitness': best_fitness
                    }

            # 保存当前结果的训练数， 模型state和optimizer的state
            torch.save(ckpt, cfg.save_dir + cfg.name + f'v5s_offical_model_{e + 1}_best.pt')

            del ckpt

        if fi > best_fitness or e % 3 == 0 or (e + 1) == cfg.epochs:
            pbar = enumerate(val_loader)
            pbar = tqdm(pbar, total=len(val_loader))
            # 保存结果
            for _, (images, targets, _, _) in pbar:
                pbar.desc = 'saving images ... ... '
                with torch.no_grad():
                    model.eval()
                    imgs = images.cuda().float() / 255.0 if cfg.use_gpu else images.float() / 255.0
                    preds, _ = model(imgs)  # torch.Size([bs, 25200, 85])
                    for i in range(imgs.shape[0]):  # 改为imgs.shape[0]， 因为这里loader没启用drop last，所以不能用batch size作为循环的次数
                        f = cfg.save_dir + cfg.name + cfg.save_image_dir + f'v5_official_batch_{e}_{i}'
                        # 从batch 中取出一个image
                        image = imgs[i] * 255

                        #  从targets中取出一个target
                        target = [x for x in targets if x[0] == i]
                        # tensor([   0.0000,   23.0000,  585.5040,  459.5063,    0.1703,    0.5640])]

                        pred = preds[i]  # pred 中 x y w h obj 80个cls  共85维，其中obj表示是否含有目标
                        pre = pred[:, 4].cpu().numpy()
                        top_indices = pre.argsort()[-20:]
                        pred = pred[top_indices]
                        cls = pred[:, 5:]
                        cls = cls.argmax(1)
                        pos = pred[:, :4]
                        pos[:, :] /= cfg.img_size
                        if cfg.use_gpu:
                            pr = torch.cat([torch.zeros(20, 1).cuda(), cls.unsqueeze(1).float(), pos], 1)
                        else:
                            pr = torch.cat([torch.zeros(20, 1), cls.unsqueeze(1).float(), pos], 1)

                        #            print(image.shape)  # torch.Size([4, 3, 640, 640]) image就是0-255的，所以不用进行操作
                        save_box_img(image, target, image.shape[2], image.shape[1], prefix=f + '_gt.jpg',
                                     names=dataparams.get('names'))
                        save_box_img(image, pr, image.shape[2], image.shape[1], prefix=f + '_pred.jpg',
                                     names=dataparams.get('names'))

# end epoch
