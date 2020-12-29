# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil
from cv2 import cv2

from utils import readlines
from kitti_utils import generate_depth_map
from tqdm import tqdm


def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark", "completion"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    side_map = {"2": '2', "3": '3', "l": '2', "r": '3'}
    t = tqdm(total=len(lines))
    for line in lines:

        folder, frame_id, side = line.split()
        frame_id = int(frame_id)

        gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                     "groundtruth", "image_0" + side_map[side],
                                     "{:010d}.png".format(frame_id))
        gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_ANYDEPTH)
        gt_depth = gt_depth.astype(np.float32) / 256

        gt_depths.append(gt_depth.astype(np.float32))
        t.update()

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()
