from torchvision import transforms
from .randaugment import RandAugment

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

min_scale = 0.2
max_scale = 1

size = 224

def get_pre_transform():
    tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(min_scale, max_scale)),
    ])
    return tf

def get_final_transform():
    tf = transforms.Compose([
        transforms.Normalize(mean=mean, std=std),
        transforms.ToTensor(),
    ])
    return tf

def get_multi_transform():
    w_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size, padding=int(0.125 * size), padding_mode='reflect'),
    ])
    s_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size, padding=int(0.125 * size), padding_mode='reflect'),
        RandAugment(2, 10)
    ])
    augs = {
        'weak': w_aug,
        'strong': s_aug
    }
    return augs