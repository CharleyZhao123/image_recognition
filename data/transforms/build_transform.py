import torchvision.transforms as transforms


def build_transform(cfg, period='train'):
    assert period in [
        'train', 'test'
    ], 'Transform Error: period {} is not defined!'.format(period)
    # Resize
    input_size = cfg.INPUT.SIZE_TRAIN  # [384, 128]

    # Values for image normalization
    pixel_mean = cfg.INPUT.PIXEL_MEAN  # [0.485, 0.456, 0.406]
    pixel_std = cfg.INPUT.PIXEL_STD  # [0.229, 0.224, 0.225]

    # Values for data augmentation
    horizontal_flip_probability = cfg.INPUT.FLIP_PROB  # 0.5
    padding_size = cfg.INPUT.IMG_PADDING  # 10
    normalize_transform = transforms.Normalize(mean=pixel_mean, std=pixel_std)
    if period is 'train':
        transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            # transforms.RandomRotation(20),
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(horizontal_flip_probability),
            # transforms.Pad(padding_size),
            # transforms.RandomCrop(input_size),
            transforms.ToTensor(), normalize_transform,
        ])
    else:
        transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize(input_size),
            transforms.ToTensor(), normalize_transform,
        ])

    return transform
