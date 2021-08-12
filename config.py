class config:
    def __init__(self):
        self.batch_size = 4
        self.cuda = False
        self.dataset = 'voc'
        self.debug = False
        self.ema = False
        self.eval_epoch = 1
        self.gamma = 0.1
        self.high_resolution = False
        self.lr = 0.0001
        self.momentum = 0.9
        self.mosaic = False
        self.multi_scale = False
        self.no_warm_up = True  # True 不使用warm up 学习率
        self.num_workers = 8
        self.resume = None
        self.save_folder = 'result/model/'
        self.start_epoch = 0
        self.tfboard = False
        self.version = 'yolov4'
        self.weight_decay = 0.0005
        self.wp_epoch = 2
        self.max_epoch = 2
        self.class_num = 20
        self.voc_root = 'G:/python/recreate/data/voc_2007_trainval/VOCdevkit/'
        self.test_root = 'G:/python/recreate/data/voc_2007_test/VOCdevkit/'
        self.classes = (  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        self.train_img_size = 640
        self.val_img_size = 416
        self.anchor_size = []
        self.ignore_threshold = 0.5
        self.pretrain = False


class configv5:
    def __init__(self):
        self.class_num = 20
        self.wp_epoch = 2
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.batch_size = 8
        self.train_img_size = 320  # 原大小640， 为了节约计算成本改为320
        self.voc_root = 'E:/data/'
        self.mosaic = False
        self.resume = None
        self.lr = 0.0001
        self.start_epoch = 0
        self.max_epoch = 2
        self.stride = [8, 16, 32]
        self.train_img_size = 320
        self.anchor_size_voc = [[32.64, 47.68], [50.24, 108.16], [126.72, 96.32],
                                [78.4, 201.92], [178.24, 178.56], [129.6, 294.72],
                                [331.84, 194.56], [227.84, 325.76], [365.44, 358.72]]
        self.random_size_range = [10, 19]
        self.ignore_thresh = .5
        self.multi_scale = False
        self.version = 'v5'
        self.cuda = False
        self.scale_factor = 100.
        self.pretrained = True
        self.pretrained_path = 'result/model/yolov5s.pt'


conf = config()
