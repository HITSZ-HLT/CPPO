#!/bin/bash

Time=$(date "+%Y%m%d-%H%M%S")

log_file=log_dir/$1_${Time}.log
CUDA_VISIBLE_DEVICES=$2 accelerate launch --main_process_port $3 --config_file configs/default_accelerate_config.yaml $1.py > $log_file 2>&1 &
echo $log_file
