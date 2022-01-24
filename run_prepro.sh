#!/bin/bash

FILE=$1
echo "run_prepro.sh running..."
echo "file: $FILE"
which python
HOST_NAME=$(/bin/hostname -s)
echo "HOST_NAME: $HOST_NAME"

echo "MASTER_ADDR" $MASTER_ADDR
echo "MASTER_PORT" $MASTER_PORT
echo "NPROC_PER_NODE" $NPROC_PER_NODE
echo "SLURM_JOB_NAME" $SLURM_JOB_NAME
echo "SLURM_JOB_ID" $SLURM_JOB_ID
echo "SLURM_JOB_NODELIST" $SLURM_JOB_NODELIST
echo "SLURM_JOB_NUM_NODES" $SLURM_JOB_NUM_NODES
echo "SLURM_LOCALID" $SLURM_LOCALID
echo "SLURM_NODEID" $SLURM_NODEID
echo "SLURM_PROCID" $SLURM_PROCID

 run_cmd="python prepare_data.py \
    --tokenizer_path oscar+wiki.64k.wordpiece.tokenizer.json \
    --train_file $FILE \
    --validation_split_percentage 2 \
    --max_seq_length 512 \
    --preprocessing_num_workers $SLURM_CPUS_PER_TASK \
    --output_dir ./hface_model_out \
    --tokenized_and_grouped_data ./data/tok_and_group_data \
    --line_by_line true \
    --config_name model_config.json \
    --model_type bert \
    --cache_dir ./hface_cache \
    --overwrite_cache false \
    --do_train true \
    --do_eval true \
    --save_total_limit 5 \
    --fp16 false \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 16
    "
    # --max_train_samples null \
    # --max_eval_samples null \
echo $run_cmd

$run_cmd

echo "done"

exit 0