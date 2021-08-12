# -*- coding: utf-8 -*-
import cv2
import numpy as np
from config import *


def save_box_img(images, targets, input_size, index2=1):
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    font = cv2.FONT_HERSHEY_SIMPLEX

    img = images[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    cv2.imwrite('result/images/'+f'{index2}_ori.jpg', img)

    img_ = cv2.imread('result/images/'+f'{index2}_ori.jpg')
    for box in targets:
        xmin, ymin, xmax, ymax = box[:-1]
        label = int(box[-1])
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        cv2.putText(img_, conf.classes[label], (int(xmin), int(ymin)), font, 1, (255, 255, 255), 1)
        
        
    cv2.imwrite('result/images/'+f'{index2}_box.jpg', img_)
