import os
import time
import torch
import logging
import torch.cuda
import numpy as np
import torch.nn as nn
import csv
from utils import AverageMeter
from utils import generate_result


def do_train(cfg, model, train_loader, val_loader,
             optimizer, scheduler, loss_fn, experiment_name):
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = os.path.join(cfg.MODEL.OUTPUT_PATH, experiment_name)

    logger = logging.getLogger('{}.train'.format(cfg.PROJECT.NAME))
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(
                torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    val_acc_meter = AverageMeter()

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss_meter.reset()
        train_acc_meter.reset()
        val_loss_meter.reset()
        val_acc_meter.reset()

        scheduler.step()

        model.train()
        one_epoch_iterations = 0
        for iteration, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            vid = torch.tensor(vid)
            target = vid.to(device)

            score = model(img)
            loss = loss_fn(score, target)

            loss.backward()
            optimizer.step()

            acc = (score.max(1)[1] == target).float().mean()
            train_loss_meter.update(loss.item(), img.shape[0])
            train_acc_meter.update(acc, 1)

            if (iteration + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}/{}] Iteration[{}/{}] Train_Loss: {:.3f}, Train_Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, epochs, (iteration + 1), len(train_loader),
                            train_loss_meter.avg, train_acc_meter.avg,
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
                'Image_Classification_' + experiment_name + '_{}.pth'.format(epoch))
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % eval_period == 0:
            model.eval()
            for iteration, (img, vid) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    vid = torch.tensor(vid)
                    target = vid.to(device)
                    score = model(img)
                    loss = loss_fn(score, target)

                    acc = (score.max(1)[1] == target).float().mean()
                    val_loss_meter.update(loss.item(), img.shape[0])
                    val_acc_meter.update(acc, 1) 

                    logger.info(
                        "Epoch[{}/{}] Iteration[{}/{}] Val_Loss: {:.3f}, Val_Acc: {:.3f}"
                        .format(epoch, epochs, (iteration + 1), len(val_loader),
                                val_loss_meter.avg, val_acc_meter.avg))

def do_test(cfg, model, test_loader, experiment_name):
    test_acc_meter = AverageMeter()
    test_acc_meter.reset()

    device = cfg.MODEL.DEVICE

    logger = logging.getLogger('{}.test'.format(cfg.PROJECT.NAME))
    logger.info("Enter Image Classification Test")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for test'.format(
                torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    # generate result csv
    output_dir = os.path.join(cfg.MODEL.OUTPUT_PATH, experiment_name)
    result_path = os.path.join(
        output_dir,
        experiment_name  + '_' + 'test_result.csv')
    with open(result_path, 'w') as f:
        f.write("file_name,label,predictive_label")

    model.eval()
    for iteration, (img, vid, vname) in enumerate(test_loader):
        with torch.no_grad():
            img = img.to(device)
            vid = torch.tensor(vid)
            target = vid.to(device)            
            score = model(img)
            p_label = score.max(1)[1]
            acc = (score.max(1)[1] == target).float().mean()
            test_acc_meter.update(acc, 1)

            logger.info(
                "Iteration[{}/{}], Test_Acc: {:.3f}"
                .format((iteration + 1), len(test_loader), test_acc_meter.avg))

        with open(result_path, 'a+') as f:
            for i in range(len(vid)):
                name = list(vname)[i]
                label = str(vid[i].item())
                p_label_ = str(p_label[i].item())
                f.write('\n')
                f.write(name +','+label+ ',' +p_label_)

        # generate_result()

