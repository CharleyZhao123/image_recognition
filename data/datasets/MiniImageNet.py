import os
import json
from PIL import Image
from torch.utils.data import Dataset


class MiniImageNet(object):
    def __init__(self, cfg, dataset_type='train'):
        super(MiniImageNet, self).__init__()

        self.dataset_path = cfg.MODEL.DATA_PATH
        if dataset_type is 'train':
            self.train_path = os.path.join(self.dataset_path, 'MVB_train',
                                           'Info', 'train+val_gallery.json')
            self.val_path = os.path.join(self.dataset_path, 'MVB_train',
                                         'Image', 'val')
            self.check_before_run()

            self.train, self.train_gallery_ids = get_train_data(
                self.train_path, begin_id=cfg.MODEL.CLASS_BEGIN, end_id=cfg.MODEL.CLASS_END)

            self.val_gallery, self.val_probe, self.num_probe = get_gallery_and_probe_data(
                self.val_path)
        elif dataset_type is 'inference':
            self.inference_path = os.path.join(self.dataset_path, 'MVB_val',
                                               'Image')
            self.inference_gallery, self.inference_probe, self.num_probe = get_gallery_and_probe_data(
                self.inference_path)
        else:
            assert 'Error: dataset_type: {} is not defined!'.format(
                dataset_type)

    def check_before_run(self):
        check_file(self.dataset_path)
        check_file(self.train_path)
        check_file(self.val_path)


class ImageDataset(Dataset):
    def __init__(self,
                 cfg,
                 dataset,
                 period='train',
                 dataset_type='train',
                 transform=None):

        assert dataset_type in ['train', 'gallery', 'probe'], \
            'ImageDataset Error: dataset_type: {} is not defined!'.format(dataset_type)
        assert period in ['train', 'inference'], \
            'ImageDataset Error: period: {} is not defined!'.format(period)

        self.dataset = dataset
        self.dataset_path = cfg.MODEL.DATA_PATH
        self.transform = transform
        self.dataset_type = dataset_type
        self.period = period

        if self.period is 'train':
            image_dir = 'MVB_train/Image'
            if self.dataset_type is not 'train':
                image_dir = 'MVB_train/Image/val'
        else:
            image_dir = 'MVB_val/Image'
        self.image_dir = image_dir

    def __getitem__(self, item):
        if self.dataset_type is 'train':
            train_image_dir = os.path.join(self.image_dir, 'train+val_gallery')
            _, image_name, image_datatype, baggage_id, baggage_material = self.dataset[
                item]
            image_path = os.path.join(self.dataset_path, train_image_dir,
                                      image_name)
            image = read_image(image_path)
            if self.transform is not None:
                image = self.transform(image)

            return image, baggage_id, image_datatype, baggage_material

        elif self.dataset_type is 'gallery':
            gallery_image_dir = os.path.join(self.image_dir, 'gallery')
            image_name, image_datatype, baggage_id = self.dataset[item]
            image_path = os.path.join(self.dataset_path, gallery_image_dir,
                                      image_name)
            image = read_image(image_path)
            if self.transform is not None:
                image = self.transform(image)

            return image, baggage_id, image_datatype

        elif self.dataset_type is 'probe':
            probe_image_dir = os.path.join(self.image_dir, 'probe')
            image_name, image_datatype = self.dataset[item]
            image_path = os.path.join(self.dataset_path, probe_image_dir,
                                      image_name)
            image = read_image(image_path)

            if self.transform is not None:
                image = self.transform(image)

            if self.period is 'train':
                baggage_id = int(image_name[:4])
                return image, image_name, baggage_id, image_datatype

            return image, image_name[:-4], image_datatype

    def __len__(self):
        return len(self.dataset)


def get_train_data(json_path, begin_id, end_id):
    baggage_dataset = []
    gallery_id_list = []
    with open(json_path, 'r') as f:
        images = json.loads(f.read())["image"]
        for image in images:
            baggage_id = int(image["id"])
            if begin_id <= baggage_id <= end_id:
                baggage_id = baggage_id-begin_id 
                image_id = image["image_id"]
                image_name = image["image_name"]
                image_datatype = image["datatype"]
                baggage_material = image["material"]
                baggage_dataset.append((image_id, image_name, image_datatype,
                                        baggage_id, baggage_material))
                if image_datatype is 'g':
                    gallery_id_list.append(image_id)
        return baggage_dataset, gallery_id_list


def get_gallery_and_probe_data(path):
    gallery_path = os.path.join(path, 'gallery')
    probe_path = os.path.join(path, 'probe')
    check_file(gallery_path)
    check_file(probe_path)

    gallery_baggage_dataset = []
    probe_baggage_dataset = []

    for root, dirs, files in os.walk(gallery_path, topdown=True):
        for name in files:
            if not name[-4:] == '.jpg':
                continue
            baggage_image_info = name.split('_')
            image_name = name
            image_datatype = 'g'
            baggage_id = int(baggage_image_info[0])
            gallery_baggage_dataset.append(
                (image_name, image_datatype, baggage_id))

    for root, dirs, files in os.walk(probe_path, topdown=True):
        for name in files:
            image_name = name
            image_datatype = 'p'
            probe_baggage_dataset.append((image_name, image_datatype))

    return gallery_baggage_dataset, probe_baggage_dataset, len(
        probe_baggage_dataset)


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
    from data.transforms import build_transform

    train_mvb = MultiViewBaggage(dataset_type='train')
    inference_mvb = MultiViewBaggage(dataset_type='inference')

    train = train_mvb.train
    train_val_probe = train_mvb.train_val_probe
    train_val_gallery = train_mvb.train_val_gallery
    train_num_probe = train_mvb.num_probe

    inference_probe = inference_mvb.inference_probe
    inference_gallery = inference_mvb.inference_gallery
    inference_num_probe = inference_mvb.num_probe

    train_period = 'train'
    inference_period = 'inference'
    train_transform = build_transform(period=train_period)
    inference_transform = build_transform(period=inference_period)

    # mvb_dataset = ImageDataset(dataset=train, period=train_period, dataset_type='train', transform=train_transform)
    # mvb_dataset = ImageDataset(dataset=train_val_probe, period=train_period, dataset_type='probe', transform=inference_transform)
    # mvb_dataset = ImageDataset(dataset=train_val_gallery, period=train_period, dataset_type='gallery', transform=inference_transform)
    # mvb_dataset = ImageDataset(dataset=inference_probe, period=inference_period, dataset_type='probe', transform=inference_transform)
    mvb_dataset = ImageDataset(dataset=inference_gallery,
                               period=inference_period,
                               dataset_type='gallery',
                               transform=inference_transform)

    print(mvb_dataset.__len__())
    batch = 3
    import random
    for i in range(batch):
        index = random.randint(0, 293)
        print(mvb_dataset.__getitem__(index))
    print('done')
