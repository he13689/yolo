import torch
import torch.nn as nn
import torch.nn.functional as F

'''
自定义了一个MSEWithLogitsLoss
通过pred_conf, pred_cls, pred_txtytwth, pred_iou, label
计算置信度、分类、iou、boundbox的loss
'''


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits, targets, mask):
        inputs = self.sigmoid(logits)

        # We ignore those whose tarhets == -1.0.
        pos_id = (mask == 1.0).float()
        neg_id = (mask == 0.0).float()
        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * (inputs) ** 2
        loss = 5.0 * pos_loss + 1.0 * neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size
            return loss
        else:
            return loss


def loss(pred_conf, pred_cls, pred_txtytwth, pred_iou, label):
    # loss func
    # reduction 该参数在新版本中是为了取代size_average和reduce参数的。它共有三种选项'elementwise_mean'，'sum'和'none'。'elementwise_mean'为默认情况，表明对N个样本的loss进行求平均之后返回(相当于reduce=True，size_average=True);'sum'指对n个样本的loss求和(相当于reduce=True，size_average=False);'none'表示直接返回n分样本的loss(相当于reduce=False)

    conf_loss_function = MSEWithLogitsLoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduce=False)
    txty_loss_function = nn.BCEWithLogitsLoss(reduce=False)
    twth_loss_function = nn.MSELoss(reduce=False)
    iou_loss_function = nn.SmoothL1Loss(reduce=False)

    # pred
    pred_conf = pred_conf[:, :, 0]
    pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]
    pred_iou = pred_iou[:, :, 0]

    # gt
    gt_conf = label[:, :, 0].float()
    gt_obj = label[:, :, 1].float()
    gt_cls = label[:, :, 2].long()
    gt_txty = label[:, :, 3:5].float()
    gt_twth = label[:, :, 5:7].float()
    gt_box_scale_weight = label[:, :, 7].float()
    gt_iou = (gt_box_scale_weight > 0.).float()
    gt_mask = (gt_box_scale_weight > 0.).float()

    # print('111 ', pred_conf, pred_cls, pred_txtytwth, pred_iou.size(), label)
    # torch.Size([4, 6300]) torch.Size([4, 20, 6300]) torch.Size([4, 6300, 4]) torch.Size([4, 6300]) torch.Size([4, 6300, 8])
    # print('111 ', pred_conf, pred_cls, pred_txtytwth, pred_iou.size(), label)
    # 出现 111  tensor([[-inf., -inf., -inf.,  ..., -inf., -inf., -inf.],
    #         [-inf., -inf., -inf.,  ..., -inf., -inf., -inf.],
    #         [-inf., -inf., -inf.,  ..., -inf., -inf., -inf.],
    #         [-inf., -inf., -inf.,  ..., -inf., -inf., -inf.]], device='cuda:0')
    batch_size = pred_conf.size(0)
    # objectness loss
    conf_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)

    # class loss
    cls_loss = torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_mask) / batch_size

    # box loss
    txty_loss = torch.sum(
        torch.sum(txty_loss_function(pred_txty, gt_txty), dim=-1) * gt_box_scale_weight * gt_mask) / batch_size
    twth_loss = torch.sum(
        torch.sum(twth_loss_function(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight * gt_mask) / batch_size
    bbox_loss = txty_loss + twth_loss

    # iou loss
    iou_loss = torch.sum(iou_loss_function(pred_iou, gt_iou) * gt_mask) / batch_size

    return conf_loss, cls_loss, bbox_loss, iou_loss
