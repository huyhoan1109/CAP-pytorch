DATA_PATH = 'data'
CHECKPOINT_PATH = 'checkpoints'
LAST_MODEL = 'last.pth'
BEST_MODEL = 'best.pth'
NEG_EPSILON = 1e-5
WARMUP_EPOCH = 10
LAMBDA_U = 1
T = 1
TOTAL_EPOCH = 400
TOTAL_ITERS = 1000
LEARNING_RATE = 0.0001

OPTIMIZER = {
    1: {
        'name': 'SGD',
        'lr': LEARNING_RATE,
        'momentum': 0.001,
        'w_decay': 0,
        'nesterov': True
    },
    2: {
        'name': 'Adam',
        'lr': LEARNING_RATE,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'w_decay': 0,
        'amsgrad': True
    },
    3: {
        'name': 'AdamW',
        'lr': LEARNING_RATE,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'w_decay': 0,
        'amsgrad': True
    }
}

SCHEDULER = {
    1: {
       'name': 'StepLR', 
       'step_size': 5
    },
    2: {
       'name': 'CosineAnnealingLR',
       'eta_min': 0.0001,
       'T_max': 400
    },
    3: {
       'name': 'OneCycleLR', 
       'max_lr': LEARNING_RATE
    },
}

DATASET_INFO = {
    'voc2012': {
        'num_classes': 20,
        'root': f'{DATA_PATH}',
        'meta': f'{DATA_PATH}/metadata/voc2012',
        'images': f'{DATA_PATH}/VOCdevkit/VOC2012/JPEGImages',
        'split': f'{DATA_PATH}/VOCdevkit/VOC2012/ImageSets/Main',
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'extension': '.tar'
    },

    'cub': {
        'num_classes': 200,
        'root': f'{DATA_PATH}/cub',
        'meta': f'{DATA_PATH}/metadata/cub',
        'images': f'{DATA_PATH}/cub/CUB_200_2011/images',
        'split': f'{DATA_PATH}/cub/CUB_200_2011/train_test_split.txt',
        'url': 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz',
        'extension': '.tgz'
    },
}

CAT2ID = {
    'voc2012': {
        'aeroplane': 0,
        'bicycle': 1,
        'bird': 2,
        'boat': 3,
        'bottle': 4,
        'bus': 5,
        'car': 6,
        'cat': 7,
        'chair': 8,
        'cow': 9,
        'diningtable': 10,
        'dog': 11,
        'horse': 12,
        'motorbike': 13,
        'person': 14,
        'pottedplant': 15,
        'sheep': 16,
        'sofa': 17,
        'train': 18,
        'tvmonitor': 19
    }
}

ID2CAT = {
    key: {
        CAT2ID[key][i]: i for i in CAT2ID[key]
    } for key in CAT2ID.keys()
}

CHUNK_SIZE = 1024

MAX_ESTOP = 5