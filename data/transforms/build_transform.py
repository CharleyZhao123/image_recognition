from .random_erasing import RandomErasing
from .rotating import Rotating
import torchvision.transforms as transforms


def build_transform(cfg, period='train'):
    assert period in [
        'train', 'inference'
    ], 'Transform Error: period {} is not defined!'.format(period)
    # Resize
    input_size = cfg.INPUT.SIZE_TRAIN  # [384, 128]

    # Values for image normalization
    pixel_mean = cfg.INPUT.PIXEL_MEAN  # [0.485, 0.456, 0.406]
    pixel_std = cfg.INPUT.PIXEL_STD  # [0.229, 0.224, 0.225]

    # Values for data augmentation
    horizontal_flip_probability = cfg.INPUT.FLIP_PROB  # 0.5
    erasing_probability = cfg.INPUT.ERASE_PROB  # 0.5
    rotating_probability = cfg.INPUT.ROTATE_PROB  # 0.5
    padding_size = cfg.INPUT.IMG_PADDING  # 10

    normalize_transform = transforms.Normalize(mean=pixel_mean, std=pixel_std)
    if period is 'train':
        transform = transforms.Compose([
#            Rotating(probability=rotating_probability),
            transforms.RandomRotation(20),
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(horizontal_flip_probability),
            transforms.Pad(padding_size),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(), normalize_transform,
            RandomErasing(probability=erasing_probability, mean=pixel_mean)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(), normalize_transform
        ])

    return transform
