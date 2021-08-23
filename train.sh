#!/usr/bin/bash
# USAGE: bash $0

DATE=`date +%Y%m%d-%H%M`

END="\e[m"
GREEN="\e[32m"
BLUE="\e[34m"
YELLOW="\e[33m"

LOG_DIR=logs
MODEL_DIR=models/outputs
TRAIN_FILE=data/twitter.128len.db
EVAL_FILE=data/dummy_data.tsv

mkdir -p $LOG_DIR $MODEL_OUTPUT

# Training on 8 GPUs
# python -m torch.distributed.launch --nproc_per_node=8 ./LSP_train.py
CUDA_VISIBLE_DEVICES=0 python LSP_train_ja.py \
  --model_name_or_path rinna/japanese-gpt2-medium \
  --init_checkpoint "None" \
  --train_input_file $TRAIN_FILE \
  --eval_input_file $EVAL_FILE \
  --output_dir $MODEL_DIR \
  --seed 42 \
  --max_seq_length 128 \
  --train_batch_size 128 \
  --gradient_accumulation_steps 8 \
  --eval_batch_size 64 \
  --learning_rate 1e-5 \
  --num_optim_steps 10000 \
  --valid_step 5000 \
  --warmup_steps 4000 \
  --normalize_data true \
  --fp16 true \
  --lr_schedule noam \
  --loss_scale 0.0 \
  --no_token_id true \
  --pbar true \
| tee $LOG_DIR/train_${DATE}.log
