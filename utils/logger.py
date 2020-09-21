import os
import sys
import datetime
import logging


def setup_logger(name, save_dir, experiment_name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    now_time = datetime.datetime.now()
    now_time = str(now_time)[:19]
    log_name = experiment_name + '_' + now_time + '_log.txt'
    log_path = os.path.join(save_dir, log_name)

    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, log_path
