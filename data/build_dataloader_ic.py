import torch
from torch.utils.data import DataLoader
from data.transforms import build_transform
from data.datasets import IC, ImageDataset

from data.samplers import RandomIdentitySampler

def collate_fn(batch):
    images, image_ids = zip(*batch)
    return torch.stack(images, dim=0), image_ids

def collate_fn_test(batch):
    images, image_ids, image_names = zip(*batch)
    return torch.stack(images, dim=0), image_ids, image_names

# 5 types of dataloader: train:[train, gallery, probe]; inference:[gallery, probe]
def build_dataloader_ic(cfg, period='train', loader_type='train'):
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
        train_ic = IC(cfg=cfg, dataset_type='train')
        val_ic = IC(cfg=cfg, dataset_type='val')

        if loader_type is 'train':
            train = train_ic.train
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
            val = val_ic.val
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
        test_ic = IC(cfg=cfg, dataset_type='test')
        test = test_ic.test
        test_set = ImageDataset(cfg=cfg,
                                dataset=test,
                                period=test_period,
                                dataset_type='test',
                                transform=test_transform)
        test_loader = DataLoader(test_set,
                                 batch_size=test_batch,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn = collate_fn_test)
        return test_loader


if __name__ == '__main__':
    pass
