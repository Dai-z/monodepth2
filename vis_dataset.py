"""Visualize generated dataset
"""
import argparse
import os
from os.path import join, isdir

from cv2 import cv2
import numpy as np
import torch

from utils import vis_depth
from layers import disp_to_depth


def main():
    """Main function
    """
    # check path & prepare file
    if isdir(args.data_path):
        rgb_dir = join(args.data_path, 'rgb')
        depth_dir = join(args.data_path, 'groundtruth')
    if isdir(args.rgb_path):
        rgb_dir = args.rgb_path
    if isdir(args.depth_path):
        depth_dir = args.depth_path
    if not isdir(rgb_dir):
        print('rgb dir:', rgb_dir, 'not exist!')
    if not isdir(depth_dir):
        print('depth dir:', depth_dir, 'not exist!')
    rgb_file_names = os.listdir(rgb_dir)
    rgb_file_names.sort()
    depth_file_names = os.listdir(depth_dir)
    depth_file_names.sort()
    rgb_file_names = list(map(lambda x: join(rgb_dir, x), rgb_file_names))
    depth_file_names = list(map(lambda x: join(depth_dir, x), depth_file_names))
    if args.save:
        save_path = rgb_dir.rstrip('/').rsplit('/', 1)[0] + '/vis'
        if not isdir(save_path):
            os.makedirs(save_path)
        print(save_path)

    print('Visualizing depth data in: ', depth_dir)
    max_idx = min(len(rgb_file_names), len(depth_file_names))
    idx = args.start_idx
    while True:
        if idx < args.start_idx:
            continue
        image = cv2.imread(rgb_file_names[idx])
        image = cv2.putText(image, str(idx), (100, 100),
                            cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)
        if args.disp:
            disp = np.load(depth_file_names[idx])
            disp = torch.from_numpy(disp)
            disp = torch.nn.functional.interpolate(
                disp, (image.shape[0], image.shape[1]),
                mode="bilinear",
                align_corners=False)
            max_num = np.percentile(disp, 95)
            min_num = np.percentile(disp, 5)
            disp[disp > max_num] = max_num
            disp[disp < min_num] = min_num
            disp -= min_num
            disp /= (max_num - min_num)
            depth = 15. / (1e-3 + disp.squeeze().numpy())
        else:
            depth = cv2.imread(depth_file_names[idx], cv2.IMREAD_ANYDEPTH)

        vis = vis_depth(depth, image, args.vis_type, scale=100, wait_key=1)

        if args.save:
            cv2.imwrite(join(save_path, rgb_file_names[idx].split('/')[-1]),
                        vis)
        key = cv2.waitKey(0)
        if key == 100:
            idx += 1
            idx = min(max_idx - 1, idx)
        elif key == 97:
            idx -= 1
            idx = max(0, idx)
        elif key == 113:
            cv2.destroyAllWindows()
            exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='',
                        help='Path of dataset to visualize')
    # specify rgb & depth data manually
    parser.add_argument('--depth_path',
                        type=str,
                        default='',
                        help='Path of depth data to visualize')
    parser.add_argument('--rgb_path',
                        type=str,
                        default='',
                        help='Path of rgb data to visualize')
    parser.add_argument('--vis_type',
                        type=str,
                        choices=['mask', 'both'],
                        default='mask',
                        help='visualize type.[mask]')
    parser.add_argument('--start_idx',
                        type=int,
                        default=0,
                        help='start from index [0]')
    parser.add_argument('--disp',
                        action='store_true',
                        help='set to visualize disparity')
    parser.add_argument('--save',
                        action='store_true',
                        help='set to save visualize image')
    args = parser.parse_args()

    main()
