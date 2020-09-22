from yacs.config import CfgNode

_C = CfgNode()

_C.PROJECT = CfgNode()
_C.PROJECT.NAME = 'image_classification'
_C.PROJECT.PATH = '/space1/zhaoqing/code/image_recognition'

##############################################################################
# -------------------------- model configuration --------------------------- #
##############################################################################
_C.MODEL = CfgNode()
_C.MODEL.DEVICE = "cuda"  # Using cuda or cpu for training
_C.MODEL.DEVICE_ID = "3"  # ID number of GPU
_C.MODEL.NUM_CLASSES = 5  # classification number
_C.MODEL.NAME = 'resnet101'  # Name of backbone
_C.MODEL.LAST_STRIDE = 1  # Last stride of backbone
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'  # Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_PATH = '/space1/home/chenyanxian/.torch/models/resnet101-5d3b4d8f.pth'  # Path to pretrained model
_C.MODEL.LOSS_TYPE = [
    'default'
]
_C.MODEL.DATA_PATH = '/space1/zhaoqing/dataset/fsl/mini-imagenet'  # Path to dataset
_C.MODEL.OUTPUT_PATH = '/space1/zhaoqing/ex/image_classification_ex'
_C.MODEL.SAMPLER = 'random'
_C.MODEL.DATALOADER_NUM_WORKERS = 8

##############################################################################
# -------------------------- input configuration --------------------------- #
##############################################################################
_C.INPUT = CfgNode()
_C.INPUT.SIZE_TRAIN = [256, 128]  # Size of the image during training
_C.INPUT.SIZE_TEST = [256, 128]  # Size of the image during test
_C.INPUT.PIXEL_MEAN = [0.485, 0.456,
                       0.406]  # Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224,
                      0.225]  # Values to be used for image normalization
_C.INPUT.FLIP_PROB = 0.5  # Random probability for image horizontal flip
_C.INPUT.ERASE_PROB = 0.5  # Random probability for random erasing
_C.INPUT.IMG_PADDING = 10  # Value of padding size

##############################################################################
# -------------------------- solver configuration -------------------------- #
##############################################################################
_C.SOLVER = CfgNode()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.BASE_LR = 3e-4  # Base learning rate
_C.SOLVER.BIAS_LR_FACTOR = 2  # Factor of learning bias
_C.SOLVER.MOMENTUM = 0.9  # SGD Momentum
_C.SOLVER.GAMMA = 0.1  # decay rate of learning rate
_C.SOLVER.STEPS = (30, 55)  # decay step of learning rate
_C.SOLVER.ID_LOSS_WEIGHT = 0.5  # Balanced weight of id loss

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

# Settings of warmup
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3  # warm up factor
_C.SOLVER.WARMUP_ITERS = 10  # iterations of warm up
_C.SOLVER.WARMUP_METHOD = "linear"  # method of warm up, option: 'constant','linear'

_C.SOLVER.CHECKPOINT_PERIOD = 40  # epoch number of saving checkpoints
_C.SOLVER.LOG_PERIOD = 20  # iteration of display training log
_C.SOLVER.EVAL_PERIOD = 5  # epoch number of validation
_C.SOLVER.IMS_PER_BATCH = 80  # Number of images per batch

##############################################################################
# --------------------------- test configuration --------------------------- #
##############################################################################
_C.TEST = CfgNode()
_C.TEST.IMS_PER_BATCH = 128  # Number of images per batch during test
_C.TEST.WEIGHT = "/space1/home/chenyanxian/RemoteProject/MVB_Reid/MVB_Reid_experiment_1_120.pth"  # Path to trained model
_C.TEST.FEAT_NORM = 'yes'  # Whether feature is normalized before test if yes, it is equivalent to cosine distance


def get_configuration():
    return _C.clone()
