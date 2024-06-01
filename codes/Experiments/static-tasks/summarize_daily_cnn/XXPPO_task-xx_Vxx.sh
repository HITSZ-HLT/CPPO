#!/bin/bash

Time=$(date "+%Y%m%d-%H%M%S")
log_file=log_dir/$1_${Time}.log
CUDA_VISIBLE_DEVICES=$2 python $1.py > $log_file 2>&1 &
echo $log_file
