import os
import argparse
import sys
sys.path.append('.')
import numpy as np
import torch.cuda
from model import build_model
from data import build_dataloader
from utils import reid_metric, fuse_score, re_ranking
from utils.reid_eval import eval_func
from utils.logger import setup_logger
from config import cfg
from torch.backends import cudnn


def generate_reid_cache(config, logger):
    device = config.MODEL.DEVICE

    feat_norm = config.TEST.FEAT_NORM

    model = build_model(cfg=config)
    model.load_param(config.TEST.WEIGHT)

    if device:
        if torch.cuda.device_count() > 1:
            logger.info('Using {} GPUs for inference'.format(
                torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    # dataloader for inference
    val_period = 'train'
    val_gallery_loader = build_dataloader(cfg=config,
                                          period=val_period,
                                          loader_type='gallery')
    val_probe_loader, inference_num_probe = build_dataloader(
        cfg=config, period=val_period, loader_type='probe')

    probe_features = []
    probe_baggage_ids = []
    probe_image_names = []
    gallery_features = []
    gallery_baggage_ids = []

    for iteration, (img, baggage_id) in enumerate(val_gallery_loader):
        with torch.no_grad():
            img = img.to(device)
            val_gallery_feature = model(img)
            gallery_features.append(val_gallery_feature)
            gallery_baggage_ids.extend(np.asarray(baggage_id))

    for iteration, (img, image_name,
                    baggage_id) in enumerate(val_probe_loader):
        with torch.no_grad():
            img = img.to(device)
            val_probe_feature = model(img)
            probe_features.append(val_probe_feature)
            probe_baggage_ids.extend(np.asarray(baggage_id))
            probe_image_names.extend(image_name)

    probe_features = torch.cat(probe_features, dim=0)
    gallery_features = torch.cat(gallery_features, dim=0)
    if feat_norm == 'yes':
        logger.info("The test feature is normalized")
        probe_features = torch.nn.functional.normalize(probe_features,
                                                       dim=1,
                                                       p=2)  # along channel
        gallery_features = torch.nn.functional.normalize(gallery_features,
                                                         dim=1,
                                                         p=2)  # along channel

    probe_baggage_ids = np.asarray(probe_baggage_ids)
    gallery_baggage_ids = np.asarray(gallery_baggage_ids)

    if cfg.TEST.RE_RANKING == 'no':
        logger.info("No re-ranking")
        cosine_distance_matrix = reid_metric(probe_features,
                                             gallery_features,
                                             mode='cosine')
        euclidean_distance_matrix = reid_metric(probe_features,
                                                gallery_features,
                                                mode='euclidean')
    elif cfg.TEST.RE_RANKING == 'yes':
        logger.info("Using re-ranking")
        cosine_distance_matrix = re_ranking(probe_features,
                                            gallery_features,
                                            mode='cosine')
        euclidean_distance_matrix = re_ranking(probe_features,
                                               gallery_features,
                                               mode='euclidean')
    else:
        raise Exception("Invalid pot for re-ranking:", cfg.TEST.RE_RANKING)

    return probe_baggage_ids, gallery_baggage_ids, cosine_distance_matrix, euclidean_distance_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MVB ReID Inference and Evaluation")
    parser.add_argument("--config_file",
                        default="",
                        help="path to inference config file",
                        type=str)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    experiment_name = 'no_config.inference_and_evaluation'
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
        experiment_name = args.config_file.split('/')[-1].split(
            '.')[0] + '.inference_and_evaluation'
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger, log_path = setup_logger('{}'.format(cfg.PROJECT.NAME),
                                    cfg.MODEL.OUTPUT_PATH, experiment_name)
    logger.info("Running with config:\n{}".format(cfg.PROJECT.NAME))

    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    # TODO: check if this speeds up testing
    cudnn.benchmark = True

    probe_baggage_ids, gallery_baggage_ids, cosine_distance_matrix, euclidean_distance_matrix = generate_reid_cache(
        config=cfg, logger=logger)
    logger.info("~~~~ Use cosine distance ~~~~")
    eval_func(cosine_distance_matrix,
              probe_baggage_ids,
              gallery_baggage_ids,
              logger=logger)
    logger.info("~~~~ Use euclidean distance ~~~~")
    eval_func(euclidean_distance_matrix,
              probe_baggage_ids,
              gallery_baggage_ids,
              logger=logger)
