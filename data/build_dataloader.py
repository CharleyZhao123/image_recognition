import torch
from torch.utils.data import DataLoader
from data.transforms import build_transform
from data.datasets import MiniClassification, ImageDataset

from data.samplers import RandomIdentitySampler

def collate_fn(batch):
    images, image_ids = zip(*batch)
    return torch.stack(images, dim=0), image_ids

# 5 types of dataloader: train:[train, gallery, probe]; inference:[gallery, probe]
def build_dataloader(cfg, period='train', loader_type='train'):
    assert loader_type in ['train', 'val', 'test'], \
        'Dataloader Error: loader_type: {} is not defined!'.format(loader_type)
    assert period in ['train', 'test'], \
        'Dataloader Error: period: {} is not defined!'.format(period)

    num_workers = cfg.MODEL.DATALOADER_NUM_WORKERS
    train_batch = cfg.SOLVER.IMS_PER_BATCH
    test_batch = cfg.TEST.IMS_PER_BATCH

    train_period = 'train'
    test_period = 'test'
    train_transform = build_transform(cfg=cfg, period=train_period)
    test_transform = build_transform(cfg=cfg, period=test_period)

    if period is 'train':
        train_mini = MiniClassification(cfg=cfg, dataset_type='train')
        val_mini = MiniClassification(cfg=cfg, dataset_type='val')

        if loader_type is 'train':
            train = train_mini.train
            train_set = ImageDataset(cfg,
                                     dataset=train,
                                     period=train_period,
                                     dataset_type='train',
                                     transform=train_transform)
            if cfg.MODEL.SAMPLER == 'triplet':
                train_loader = DataLoader(
                    train_set,
                    batch_size=train_batch,
                    shuffle=False,
                    num_workers=num_workers,
                    sampler=RandomIdentitySampler(train, train_batch,
                                                  cfg.INPUT.NUM_IMG_PER_ID),
                    drop_last=True)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=train_batch,
                                          shuffle=True,
                                          num_workers=num_workers,
                                          collate_fn = collate_fn)
            return train_loader

        else:
            val = val_mini.val
            val_set = ImageDataset(cfg=cfg,
                                   dataset=val,
                                   period=train_period,
                                   dataset_type='val',
                                   transform=test_transform)
            val_loader = DataLoader(val_set,
                                    batch_size=test_batch,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn = collate_fn)
            return val_loader

    else:
        inference_mvb = MultiViewBaggage(cfg=cfg, dataset_type='inference')

        if loader_type is 'gallery':
            inference_gallery = inference_mvb.inference_gallery
            inference_gallery_set = ImageDataset(cfg=cfg,
                                                 dataset=inference_gallery,
                                                 period=inference_period,
                                                 dataset_type='gallery',
                                                 transform=inference_transform)
            inference_gallery_loader = DataLoader(
                inference_gallery_set,
                batch_size=test_batch,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=gallery_collate_fn)
            return inference_gallery_loader

        else:
            inference_num_probe = inference_mvb.num_probe
            inference_probe = inference_mvb.inference_probe
            inference_probe_set = ImageDataset(cfg=cfg,
                                               dataset=inference_probe,
                                               period=inference_period,
                                               dataset_type='probe',
                                               transform=inference_transform)
            inference_probe_loader = DataLoader(
                inference_probe_set,
                batch_size=test_batch,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=inference_probe_collate_fn)
            return inference_probe_loader, inference_num_probe


if __name__ == '__main__':
    # train_period = 'train'
    # inference_period = 'inference'

    # dataloader for training
    # train_loader = build_dataloader(period=train_period, loader_type='train')
    # gallery_loader = build_dataloader(period=train_period, loader_type='gallery')
    # probe_loader, train_val_num_probe = build_dataloader(period=train_period, loader_type='probe')

    # dataloader for inference
    # inference_gallery_loader = build_dataloader(period=inference_period, loader_type='gallery')
    # inference_probe_loader, inference_num_probe = build_dataloader(period=inference_period, loader_type='probe')
    #
    # for iteration, data in enumerate(inference_probe_loader):
    #     print(iteration)
    #     print(data)
    #     print('----------------------')
    pass
