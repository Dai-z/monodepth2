#!/bin/bash

TEST=true
if [ $TEST = true ] ; then
    GPUS='7'
    ARGS="--model_name test_parallel --num_workers 1 --load_weights_folder ~/tmp/mdp/models/weights_19"
else
    GPUS='0,1,2,3,4,5,6'
    ARGS="--num_workers 120"
fi

NUM_GPUS=$(echo $GPUS | tr "," "\n" | wc -l)
NUM_THREADS=$(cat /proc/cpuinfo | grep processor | wc -l)

export CUDA_VISIBLE_DEVICES=$GPUS
export OMP_NUM_THREADS=$(($NUM_THREADS/$NUM_GPUS))


python -m torch.distributed.launch \
 --nproc_per_node=$NUM_GPUS \
 --node_rank=0 \
 --master_port=2333 \
 train.py \
 --gpus $GPUS \
 --png $ARGS


#  --model_name test_parallel \
#  --num_workers 1 \"
#  --load_weights_folder models/mono_640x192 \
#  --load_weights_folder ~/tmp/mdp/models/weights_19 \

kill -9 $(ps aux | grep train.py | grep -v grep | awk '{print $2}')