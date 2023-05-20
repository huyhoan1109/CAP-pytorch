DATA_PATH = 'data'
CHECKPOINT_PATH = 'checkpoints'
LAST_MODEL = 'last.pth'
BEST_MODEL = 'best.pth'

CHUNK_SIZE = 1024

DATASET_INFO = {
    'voc2012': {
        'root': f'{DATA_PATH}',
        'meta': f'{DATA_PATH}/metadata/voc2012',
        'images': f'{DATA_PATH}/VOCdevkit/VOC2012/JPEGImages',
        'labels': f'{DATA_PATH}/VOCdevkit/VOC2012/ImageSets/Main',
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'extension': '.tar'
        #'hash': {

        # }
    },
    'cub': {
        'root': f'{DATA_PATH}/cub',
        'meta': f'{DATA_PATH}/metadata/cub',
        'images': f'{DATA_PATH}/cub/CUB_200_2011/images',
        'labels': f'{DATA_PATH}/cub/CUB_200_2011/image_class_labels.txt',
        'url': "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
        'extension': '.tgz'
        #'hash': {

        # }
    }
}

LABEL2ID = {
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

ID2LABEL = {
    key: {
        LABEL2ID[key][i]: i for i in LABEL2ID[key]
    } for key in LABEL2ID.keys()
}