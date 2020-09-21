import os
import torch
import argparse
import sys
sys.path.append('.')
from config import cfg
from loss import build_loss
from model import build_model
from torch.backends import cudnn
from data import build_dataloader
from utils.logger import setup_logger
from engine.inference import inference
from utils.plot_curve import plot_curve
from engine.model_engine import do_train
from solver import build_optimizer, WarmUpMultiStepLR


def train(config, experiment_name=None):
    num_classes = config.MODEL.NUM_CLASSES
    class_begin = config.MODEL.CLASS_BEGIN
    class_end = config.MODEL.CLASS_END

    if num_classes != class_end - class_begin + 1:
        raise Exception(" NUM_CLASSES is not equal to ( CLASS_END - CLASS_BEGIN +1 )")

    # dataloader for training
    train_period = 'train'
    train_loader = build_dataloader(cfg=config,
                                    period=train_period,
                                    loader_type='train')
    val_gallery_loader = build_dataloader(cfg=config,
                                          period=train_period,
                                          loader_type='gallery')
    val_probe_loader, _ = build_dataloader(cfg=config,
                                           period=train_period,
                                           loader_type='probe')

    # prepare model
    model = build_model(cfg=config)

    print('Train with center loss, the loss type is',
          config.MODEL.METRIC_LOSS_TYPE)
    loss_func, center_criterion = build_loss(config, num_classes)
    optimizer, optimizer_center = build_optimizer(config, model,
                                                  center_criterion)

    # Add for using self trained model
    if config.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(
            config.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')
            [-1])
        print('Start epoch:', start_epoch)
        path_to_optimizer = config.MODEL.PRETRAIN_PATH.replace(
            'model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        path_to_center_param = config.MODEL.PRETRAIN_PATH.replace(
            'model', 'center_param')
        print('Path to the checkpoint of center_param:', path_to_center_param)
        path_to_optimizer_center = config.MODEL.PRETRAIN_PATH.replace(
            'model', 'optimizer_center')
        print('Path to the checkpoint of optimizer_center:',
              path_to_optimizer_center)
        model.load_state_dict(torch.load(config.MODEL.PRETRAIN_PATH))
        optimizer.load_state_dict(torch.load(path_to_optimizer))
        center_criterion.load_state_dict(torch.load(path_to_center_param))
        optimizer_center.load_state_dict(torch.load(path_to_optimizer_center))

    scheduler = WarmUpMultiStepLR(optimizer, config.SOLVER.STEPS,
                                  config.SOLVER.GAMMA,
                                  config.SOLVER.WARMUP_FACTOR,
                                  config.SOLVER.WARMUP_ITERS,
                                  config.SOLVER.WARMUP_METHOD)

    print('------------------ Start Training -------------------')
    do_train(config, model, center_criterion, train_loader, val_gallery_loader,
             val_probe_loader, optimizer, optimizer_center, scheduler,
             loss_func, experiment_name)
    print('---------------- Training Completed ---------------- ')


def main():
    parser = argparse.ArgumentParser(description="MVB ReID Training")
    parser.add_argument("--config_file",
                        default="",
                        help="path to config file",
                        type=str)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpu = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    experiment_name = 'no_config'
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
        experiment_name = args.config_file.split('/')[-1].split('.')[0]
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.MODEL.OUTPUT_PATH, experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger, log_path = setup_logger('{}'.format(cfg.PROJECT.NAME), output_dir,
                                    experiment_name)
    logger.info("Running with config:\n{}".format(cfg.PROJECT.NAME))

    logger.info("Using {} GPU".format(num_gpu))
    logger.info(args)
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train(cfg, experiment_name=experiment_name)

    try:
        logger.info("Drawing curve ......")
        plot_curve(log_path=log_path,
                   experiment_name=experiment_name,
                   output=output_dir)
        logger.info("The curve is saved in {}".format(output_dir))
    except Exception as e:
        print(e)


#    inference(cfg, experiment_name=experiment_name)

if __name__ == '__main__':
    main()
