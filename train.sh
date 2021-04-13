#!/bin/bash

GPUS='0'
if [[ ! -z $TEST && $TEST = true ]] ; then
    ARGS="--model_name test_parallel --num_workers 1"
else
    ARGS="--num_workers 20 "
fi
if [ -z $PORT ] ; then
    PORT=$(($RANDOM%8192+8192))
fi
if [ -z $CUDA_VISIBLE_DEVICES ] ; then
    export CUDA_VISIBLE_DEVICES=$GPUS
fi

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
# NUM_THREADS=$(cat /proc/cpuinfo | grep processor | wc -l)
echo "using GPU ["$CUDA_VISIBLE_DEVICES"] on port: "$PORT

# export OMP_NUM_THREADS=$(($NUM_THREADS/$NUM_GPUS))
export OMP_NUM_THREADS=16

python -m torch.distributed.launch \
 --nproc_per_node=$NUM_GPUS \
 --node_rank=0 \
 --master_port=$PORT \
 train.py \
 --png $ARGS $@

kill -9 $(ps aux | grep '='$PORT | grep -v grep | awk '{print $2}')