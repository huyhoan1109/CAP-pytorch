from torchvision import transforms
from .randaugment import RandAugment

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

min_scale = 0.2
max_scale = 1

def get_pre_transform():
    tf = transforms.Compose(
        transforms.RandomResizedCrop(224, scale=(min_scale, max_scale))
    )
    return tf

def get_transform():
    tf = transforms.Compose(
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    )
    return tf

def get_augmentation():
    w_aug = transforms.Compose(
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(padding=0.125, padding_mode='reflect'),
    )
    s_aug = transforms.Compose(
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(padding=0.125, padding_mode='reflect'),
        RandAugment(2, 10)
    )
    augs = {
        'weak_augmentation': w_aug,
        'strong_augmentation': s_aug
    }
    return augs