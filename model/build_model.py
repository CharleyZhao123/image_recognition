from model.ic_net import ClassificationNet


def build_model(cfg):
    num_classes = cfg.MODEL.NUM_CLASSES  # 750
    last_stride = cfg.MODEL.LAST_STRIDE  # 1 or 2
    model_name = cfg.MODEL.NAME
    pretrained_choice = cfg.MODEL.PRETRAIN_CHOICE  # 'imagenet'
    pretrained_path = cfg.MODEL.PRETRAIN_PATH  # '~/.torch/models/resnet50-19c8e357.pth'

    model = ClassificationNet(num_classes, last_stride, pretrained_path,
                 model_name, pretrained_choice)
    return model


if __name__ == '__main__':
    ic = build_model()
