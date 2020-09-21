#!/bin/bash

python engine/train.py --config_file ./experiments/resnet50.yml MODEL.DEVICE_ID "('3')" MODEL.OUTPUT_PATH "('/space1/home/lurq/code/luggage_reid/output')" 
