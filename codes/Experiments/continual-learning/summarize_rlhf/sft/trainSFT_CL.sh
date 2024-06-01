#!/bin/bash
Time=$(date "+%Y%m%d-%H%M%S")
log_file=log_dir/SFT_CL_gpt2xl_Data$1_$Time.log
CUDA_VISIBLE_DEVICES=4,5,6,7  deepspeed --master_port $2 train_sft$1_xl.py > $log_file 2>&1 &
echo $log_file