DOWNLOAD_INFO = {
    'voc2012': {
        'url': f'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'hash': '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'
    }
} 

DATA_PATH = '../data'
METADATA_PATH = '../data/metadata'
PHASES = ['train', 'val', 'test']

LABEL2ID = {
    'voc': {
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