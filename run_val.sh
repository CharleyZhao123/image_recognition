#!/bin/bash

python engine/inference_and_eval.py --config_file ./experiments/resnet50.yml TEST.WEIGHT "('./output/path_cfg.train/MVB_Reid_path_cfg.train_100.pth')"
