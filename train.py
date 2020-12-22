# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


# set OMP_NUM_THREADS = number of CPU processors/number of processes in default to neither overload or waste CPU threads

# python -m torch.distributed.launch --nproc_per_node=2 --node_rank=0 --master_port=2333 train.py
# ---args---
# nproc_per_node: num of gpus

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
