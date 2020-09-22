import os
import csv
from PIL import Image
from torch.utils.data import Dataset


class MiniClassification(object):
    def __init__(self, cfg, dataset_type='train'):
        super(MiniClassification, self).__init__()

        self.dataset_path = cfg.MODEL.DATA_PATH
        if dataset_type is 'train':
            self.train_path = os.path.join(self.dataset_path, 'mini_ic', 'ic_train.csv')
            self.train, self.train_ids = get_train_data(self.train_path)
        elif dataset_type is 'val':
            self.val_path = os.path.join(self.dataset_path, 'mini_ic', 'ic_val.csv')
            self.val, self.val_ids = get_val_data(self.val_path)
        elif dataset_type is 'test':
            self.test_path = os.path.join(self.dataset_path, 'mini_ic', 'ic_test.csv')
            self.test, self.test_id = get_test_data(self.test_path)
        else:
            assert 'Error: dataset_type: {} is not defined!'.format(
                dataset_type)

    # def check_before_run(self):
    #     check_file(self.dataset_path)
    #     check_file(self.train_path)
    #     check_file(self.val_path)
    #     check_file(self.test_path)


class ImageDataset(Dataset):
    def __init__(self,
                 cfg,
                 dataset,
                 period='train',
                 dataset_type='train',
                 transform=None):

        assert dataset_type in ['train', 'val', 'test'], \
            'ImageDataset Error: dataset_type: {} is not defined!'.format(dataset_type)
        assert period in ['train', 'test'], \
            'ImageDataset Error: period: {} is not defined!'.format(period)

        self.dataset = dataset
        self.dataset_path = cfg.MODEL.DATA_PATH
        self.transform = transform
        self.dataset_type = dataset_type
        self.period = period

        if self.period is 'train':
            image_dir = 'images'
            if self.dataset_type is not 'train':
                image_dir = 'images'
        else:
            image_dir = 'images'
        self.image_dir = image_dir

    def __getitem__(self, item):
        if self.dataset_type is 'train':
            train_image_dir = self.image_dir
            image_name, image_id = self.dataset[item]
            image_path = os.path.join(self.dataset_path, train_image_dir,
                                      image_name)
            image = read_image(image_path)
            if self.transform is not None:
                image = self.transform(image)
            return image, image_id

        elif self.dataset_type is 'val':
            val_image_dir = self.image_dir
            image_name, image_id = self.dataset[item]
            image_path = os.path.join(self.dataset_path, val_image_dir,
                                      image_name)
            image = read_image(image_path)
            if self.transform is not None:
                image = self.transform(image)

            return image, image_id

        elif self.dataset_type is 'test':
            test_image_dir = self.image_dir
            image_name, image_id = self.dataset[item]
            image_path = os.path.join(self.dataset_path, test_image_dir,
                                      image_name)
            image = read_image(image_path)

            if self.transform is not None:
                image = self.transform(image)

            return image, image_id

    def __len__(self):
        return len(self.dataset)


def get_train_data(csv_path):
    train_dataset = []
    train_name_list = []
    with open(csv_path, 'r') as f:
        data_reader = csv.reader(f)
        head_row = next(data_reader)
        for row in data_reader:
            image_name = row[0] # n0185567200000003.jpg
            image_id = int(row[1]) # n01855672->1855672
            train_dataset.append((image_name, image_id))
            train_name_list.append(image_name)
        del head_row
        return train_dataset, train_name_list

def get_val_data(csv_path):
    val_dataset = []
    val_name_list = []
    with open(csv_path, 'r') as f:
        data_reader = csv.reader(f)
        head_row = next(data_reader)
        for row in data_reader:
            image_name = row[0] # n0185567200000003.jpg
            image_id = int(row[1]) # n01855672
            val_dataset.append((image_name, image_id))
            val_name_list.append(image_name)
        del head_row
        return val_dataset, val_name_list

def get_test_data(csv_path):
    test_dataset = []
    test_name_list = []
    with open(csv_path, 'r') as f:
        data_reader = csv.reader(f)
        head_row = next(data_reader)
        for row in data_reader:
            image_name = row[0] # n0185567200000003.jpg
            image_id = int(row[1]) # n01855672
            test_dataset.append((image_name, image_id))
            test_name_list.append(image_name)
        del head_row
        return test_dataset, test_name_list

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
            return img
        except IOError:
            print(
                "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill."
                .format(img_path))


def check_file(path):
    if not os.path.exists(path):
        raise RuntimeError("'{}' is not available".format(path))


if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from transforms import build_transform

    train_mini = MiniImageNet(dataset_type='train')
    val_mini = MiniImageNet(dataset_type='val')
    test_mini = MiniImageNet(dataset_type='test')

    train = train_mini.train
    val = val_mini.val
    test = test_mini.test

    train_period = 'train'
    test_period = 'test'

    train_transform = build_transform(period=train_period)
    test_transform = build_transform(period=test_period)

    mini_dataset = ImageDataset(dataset=train,
                                period=train_period,
                                dataset_type='train',
                                transform=train_transform)

    print(mini_dataset.__len__())
    batch = 3
    import random
    for i in range(batch):
        index = random.randint(0, 293)
        print(mini_dataset.__getitem__(index))
    print('done')
