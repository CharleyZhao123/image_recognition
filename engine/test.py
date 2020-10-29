import os
import argparse
import sys
sys.path.append('.')
from config import cfg
from model import build_model
from torch.backends import cudnn
from data import build_dataloader
from utils.logger import setup_logger
from engine.model_engine import do_test


def test(config, experiment_name=None):
    # dataloader for test
    test_period = 'test'
    test_loader = build_dataloader(cfg=config,
                                   period=test_period,
                                   loader_type='test')

    # prepare model
    model = build_model(cfg=config)
    model.load_param(config.TEST.WEIGHT)

    print('------------------ Start Test -------------------')
    do_test(config, model, test_loader, experiment_name)
    print('---------------- Inference Completed -----------------')


def main():
    parser = argparse.ArgumentParser(description="Image Classification Test")
    parser.add_argument("--config_file",
                        default="",
                        help="path to test config file",
                        type=str)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpu = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    experiment_name = 'no_config.test'
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
        experiment_name = args.config_file.split('/')[-1].split(
            '.')[0] + '.test'
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.MODEL.OUTPUT_PATH, experiment_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger, log_path = setup_logger('{}'.format(cfg.PROJECT.NAME),
                                    cfg.MODEL.OUTPUT_PATH, experiment_name)
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

    test(cfg, experiment_name=experiment_name)


if __name__ == '__main__':
    main()
