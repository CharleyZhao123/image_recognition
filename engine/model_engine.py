import os
import time
import torch
import logging
import torch.cuda
import numpy as np
import torch.nn as nn
from utils import R1_mAP, AverageMeter, generate_merge_result


def do_train(cfg, model, center_criterion, train_loader, val_gallery_loader,
             val_probe_loader, optimizer, optimizer_center, scheduler, loss_fn,
             experiment_name):
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = os.path.join(cfg.MODEL.OUTPUT_PATH, experiment_name)

    embedding_metric = cfg.TEST.METHOD

    val_period = 'val'

    logger = logging.getLogger('{}.train'.format(cfg.PROJECT.NAME))
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(
                torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP(feat_norm=cfg.TEST.FEAT_NORM,
                       metric=embedding_metric,
                       period=val_period)
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        evaluator.period = val_period

        scheduler.step()

        model.train()
        one_epoch_iterations = 0
        for iteration, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            score, feat = model(img)
            loss = loss_fn(score, feat, target)

            loss.backward()
            optimizer.step()
            for param in center_criterion.parameters():
                # Increase the weight of center loss with regard to center embeddings
                param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
            optimizer_center.step()

            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (iteration + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}/{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, epochs, (iteration + 1), len(train_loader),
                            loss_meter.avg, acc_meter.avg,
                            scheduler.get_lr()[0]))
            one_epoch_iterations += 1

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (one_epoch_iterations + 1)
        logger.info(
            "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
            .format(epoch, time_per_batch,
                    train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            checkpoint_path = os.path.join(
                output_dir,
                'MVB_Reid_' + experiment_name + '_{}.pth'.format(epoch))
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % eval_period == 0:
            model.eval()
            for iteration, (img, baggage_id) in enumerate(val_gallery_loader):
                with torch.no_grad():
                    img = img.to(device)
                    val_gallery_feature = model(img)
                    evaluator.update((val_gallery_feature, baggage_id),
                                     feature_type='gallery')

            for iteration, (img, _, baggage_id) in enumerate(val_probe_loader):
                with torch.no_grad():
                    img = img.to(device)
                    val_probe_feat = model(img)
                    evaluator.update((val_probe_feat, baggage_id),
                                     feature_type='probe')

            logger = logging.getLogger('{}.val'.format(cfg.PROJECT.NAME))
            logger.info("Train Validation Results - Epoch: {}".format(epoch))
            evaluator.compute(logger)
            logger = logging.getLogger('{}.train'.format(cfg.PROJECT.NAME))


def do_inference(cfg, model, inference_gallery_loader, inference_probe_loader,
                 experiment_name):
    device = cfg.MODEL.DEVICE
    embedding_metric = cfg.TEST.METHOD
    output_dir = os.path.join(cfg.MODEL.OUTPUT_PATH, experiment_name)

    result_path = os.path.join(
        output_dir,
        experiment_name + '_' + embedding_metric + '_' + 'reid_result.csv')

    logger = logging.getLogger('{}.inference'.format(cfg.PROJECT.NAME))
    logger.info("Enter MVB Reid Inference")

    probe_features = []
    probe_image_names = []
    gallery_features = []
    gallery_baggage_ids = []

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(
                torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    for iteration, (img, baggage_id) in enumerate(inference_gallery_loader):

        with torch.no_grad():
            img = img.to(device)
            inference_gallery_feature = model(img)

            gallery_features.append(inference_gallery_feature)
            gallery_baggage_ids.extend(np.asarray(baggage_id))

    for iteration, (img, image_name) in enumerate(inference_probe_loader):
        with torch.no_grad():
            img = img.to(device)
            inference_probe_feature = model(img)

            probe_image_names.extend(image_name)
            probe_features.append(inference_probe_feature)

    generate_merge_result(cfg,
                          probe_features,
                          gallery_features,
                          probe_image_names,
                          gallery_baggage_ids,
                          metric=embedding_metric,
                          result_path=result_path)
