# ---------------------------------- 上次改错文件了， 可能存在问题需要修改------------------------------------------------

import os.path as osp
import sys
import torch
import torch.utils.data as datasets
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
from config import *


# 进行标准的一个转换， 将voc数据的标准转成 tensor里面的bb坐标和标签index
class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        # 将 classes 转为 索引 形式
        self.class_to_ind = class_to_ind or dict(zip(conf.classes, range(len(conf.classes))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : 目标标注，做成可以用的ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
            
            每个xml文件是对一张图片的说明，包括图片尺寸、名称、在图片上出现的位置等信息
        """
        res = []
        for obj in target.iter('object'):
            difficult = (int(obj.find('difficult').text) == 1) # 识别是否困难 1代表识别困难
            if not self.keep_difficult and difficult: # 是否保留difficult的图像
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                # 从 bbox 中 找到坐标
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # bndbox 的format是 [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDataset(datasets.Dataset):
    """
    VOC Detection Dataset Object
    输入图像， 输出是标注
    """

    def __init__(self,
                 root,
                 img_size=None,
                 image_sets=[('2012', 'train')],
                 transform=None,
                 base_transform=None,
                 target_transform=VOCAnnotationTransform(),
                 mosaic=False):
        self.root = root
        self.img_size = img_size or 300
        self.image_set = image_sets
        self.transform = transform  # 对图像进行transform
        self.base_transform = base_transform
        self.target_transform = target_transform
        self.mosaic = mosaic  # 马赛克
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')  # 找到所有的标注xml文件， 里面记录了图上的bb相关数据
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')  # 找到所有的图像
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)  # 数据根目录
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):  # 从trainval文件中提取所有的训练图像
                # 读取所有图片的名称 strip方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列，没有传入参数的话则去除首尾空格
                self.ids.append((rootpath, line.strip()))  # ids里面保存了rootpath 和 index 它是图片的名称 比如00005

    def __getitem__(self, item):
        img, gt, h, w = self.get_item(item)
        return img, gt

    def __len__(self):
        return len(self.ids)

    def get_item(self, index):
        img_id = self.ids[index]  # 得到xml文件所在的路径
        target = ET.parse(self._annopath % img_id).getroot()  # 解析图像的xml文件并且得到 根节点
        img = cv2.imread(self._imgpath % img_id)  # 得到jpg格式的原图像
        h, w, c = img.shape

        if self.target_transform is not None:
            # 对target进行transform
            target = self.target_transform(target, w, h)

        # 使用mosaic这种数据增强方法
        '''
        Mosaic数据增强方法是YOLOV4论文中提出来的，这种数据增强方式简单来说就是把4张图片，
        通过随机缩放、随机裁减、随机排布的方式进行拼接。根据论文的说法，
        优点是丰富了检测物体的背景和小目标，并且在计算Batch Normalization的时候一次会计算四张图片的数据，
        使得mini-batch大小不需要很大，一个GPU就可以达到比较好的效果。
        并且四张图片拼接在一起变相地提高了batch_size，在进行batch normalization的时候也会计算四张图片，
        所以对本身batch_size不是很依赖，单块GPU就可以训练YOLOV4。
        
        缺点: 如果我们的数据集本身就有很多的小目标，那么Mosaic数据增强会导致本来较小的目标变得更小，导致模型的泛化能力变差
        
        整个Mosaic过程如图一所示, 图一展示的是pleft,pright,ptop,pbot都大于0时的情况，首先在原图上找到以(pleft,ptop)为左上角
        swidth，sheight为宽和长的矩形，然后取这个矩形和原图的交集(也就是深绿色的部分)
        图1中(2)这里不是直接取的交集出来，而是先创建一个宽为swidth，长为sheight的矩形，再将矩形赋值为原图RGB三个通道的均值
        只不过图一是基于pleft,pright,ptop,pbot都大于0时的情况，所以正好放在(4)上(0, 0)坐标上。
        
        然后对图片进行resize，resize为网络输入所需要的分辨率，默认情况下就是608x608大小。
        然后根据计算的左上坐标，以及随机得到的宽CutX，长Cuty
        裁剪一部分区域作为一张新图的左上部分。图1中(4)红框表示裁剪的区域，注意：图1中(4)左上角的(0, 0)坐标是因为pleft,pright大于0，
        最后说明一下对于标签框的处理，图1中可以看到，当进行裁剪的时候，如果裁剪了样本当中的标签框的部分区域，则将其舍弃，保留裁剪之后还完整的标签框。
        '''
        if self.mosaic and np.random.randint(2):
            # np.random.randint(2) 结果是0或1
            index_list = self.ids[:index] + self.ids[index + 1:]  # 除了当前index的 图片名称
            # 每次读取四张图片。 随机抽取三个以及当前index 一共加起来4个
            id2, id3, id4 = random.sample(index_list, 3)  # 随机从index_list抽取三个
            img_list = [img]
            target_list = [target]
            for id in [id2, id3, id4]:
                mask = cv2.imread(self._imgpath % id)  # 读取随机抽取的img
                mh, mw, mc = mask.shape
                mt = ET.parse(self._annopath % id).getroot()  # 从xml中读取根节点
                mt = self.target_transform(mt, mw, mh)  # 经过transform得到target
                img_list.append(mask)
                target_list.append(mt)

            # 定义马萨克图片的，一开始是空的，大小是size*2, size*2, channel
            mosaic_img = np.zeros([self.img_size * 2, self.img_size * 2, img.shape[2]], dtype=np.uint8)
            # 马萨克图片的中心  random.uniform(x, y)方法将随机生成一个实数，它在 [x,y] 范围内。
            # -self.img_size // 2, -self.img_size // 2  两个是相等的
            yc, xc = [int(random.uniform(-x, 2 * self.img_size + x)) for x in
                      [-self.img_size // 2, -self.img_size // 2]]

            mosaic_tg = []
            for i in range(4):
                # 从list中拿到target和img path
                img_i, target_i = img_list[i], target_list[i]
                h0, w0, _ = img_i.shape  # 读取图像大小

                img_i = cv2.resize(img_i, (self.img_size, self.img_size))  # resize
                h1, w1, _ = img_i.shape

                # 根据中心
                if i == 0:  # top left
                    # xmin, ymin, xmax, ymax (large image)
                    x1a, y1a, x2a, y2a = max(xc - w1, 0), max(yc - h1, 0), xc, yc
                    # xmin, ymin, xmax, ymax (small image)
                    x1b, y1b, x2b, y2b = w1 - (x2a - x1a), h1 - (y2a - y1a), w1, h1
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h1, 0), min(xc + w1, self.img_size * 2), yc
                    x1b, y1b, x2b, y2b = 0, h1 - (y2a - y1a), min(w1, x2a - x1a), h1
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w1, 0), yc, xc, min(self.img_size * 2, yc + h1)
                    x1b, y1b, x2b, y2b = w1 - (x2a - x1a), 0, w1, min(y2a - y1a, h1)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w1, self.img_size * 2), min(self.img_size * 2, yc + h1)
                    x1b, y1b, x2b, y2b = 0, 0, min(w1, x2a - x1a), min(y2a - y1a, h1)
                else:
                    print('error!')
                    return
                # mosaic img
                mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]  # 填充mosaic图像中的指定区域
                padw = x1a - x1b
                padh = y1a - y1b

                # labels
                target_i = np.array(target_i)
                # target_i_ 里面是 bb的xywh 以及 conf置信度
                target_i_ = target_i.copy()
                if len(target_i) > 0:
                    # 取出 有效的 target，并且对它进行modify
                    # a valid target, and modify it.
                    target_i_[:, 0] = (w1 * (target_i[:, 0]) + padw)
                    target_i_[:, 1] = (h1 * (target_i[:, 1]) + padh)
                    target_i_[:, 2] = (w1 * (target_i[:, 2]) + padw)
                    target_i_[:, 3] = (h1 * (target_i[:, 3]) + padh)

                    mosaic_tg.append(target_i_)  # 把target放进mosaic_tg

            if len(mosaic_tg) == 0:
                mosaic_tg = np.zeros([1, 5])
            else:
                mosaic_tg = np.concatenate(mosaic_tg, axis=0)
                # Cutout/Clip targets 裁剪mosaic target， 把不在 0, 2 * self.img_size 这个范围的都剪掉
                np.clip(mosaic_tg[:, :4], 0, 2 * self.img_size, out=mosaic_tg[:, :4])
                # normalize
                mosaic_tg[:, :4] /= (self.img_size * 2)  # 归一化坐标，在0-1之间

            # 对mosaic图片进行transform， 得到img， box以及label
            mosaic_img, boxes, labels = self.base_transform(mosaic_img, mosaic_tg[:, :4], mosaic_tg[:, 4])
            # to rgb
            mosaic_img = mosaic_img[:, :, (2, 1, 0)]
            mosaic_tg = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # 返回图像 mosaic后target 以及图像大小
            return torch.from_numpy(mosaic_img).permute(2, 0, 1).float(), mosaic_tg, self.img_size, self.img_size
        else: # 不开启mosaic数据增强
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

            img, boxes, label = self.transform(img, target[:, :4], target[:, 4])  # 对img和target进行transform
            img = img[:, :, (2, 1, 0)]  # 变形成rgb
            target = np.hstack((boxes, np.expand_dims(label, axis=1))) # 将target stack在一起

        # 转成RGB，返回image、traget和hw
        return torch.from_numpy(img).permute(2, 0, 1).float(), target, h, w

    def get_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id

    def get_anno(self, index):
        img_id = self.ids[index]  # 得到xml文件所在的路径
        anno = ET.parse(self._annopath % img_id).getroot()  # 解析图像的xml文件并且得到 根节点
        gtbox = self.target_transform(anno, 1, 1)
        return img_id[1], gtbox  # 返回图像名称和gtbox的信息

    def get_tensor(self, index):
        return torch.Tensor(self.get_image(index)).unsqueeze(0)


if __name__ == '__main__':
    VOCDataset('E:/data/VOC07+12')
