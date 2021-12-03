import numpy as np
import os
import PIL.Image as pil
import random
import torch
from torchvision import transforms

from .mono_dataset import MonoDataset


class UAVDataset(MonoDataset):
    """Superclass for UAV dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32)

        # self.full_res_shape = (1242, 375)
        self.full_res_shape = (1024, 540)
        self.idx = list(range(self.__len__()))

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        if index == 0:
            random.shuffle(self.idx)
        inputs = {}
        index = self.idx[index]

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        frame_index = int(line[1].rsplit('.', 1)[0])
        side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i,
                        -1)] = self.get_color(folder, frame_index, other_side,
                                              do_flip)
            else:
                inputs[("color", i,
                        -1)] = self.get_color(folder, frame_index + i, side,
                                              do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2**scale)
            K[1, :] *= self.height // (2**scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(
                np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def check_depth(self):
        if len(self.filenames) == 0:
            return False
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1].rsplit('.', 1)[0])

        velo_filename = os.path.join(
            self.data_path, scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_image_path(self, folder, frame_index):
        f_str = "{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    # def get_depth(self, folder, frame_index, side, do_flip):
    #     f_str = "{:010d}.png".format(frame_index)
    #     depth_path = os.path.join(
    #         self.data_path,
    #         folder,
    #         "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
    #         f_str)

    #     depth_gt = pil.open(depth_path)
    #     depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
    #     depth_gt = np.array(depth_gt).astype(np.float32) / 256

    #     if do_flip:
    #         depth_gt = np.fliplr(depth_gt)

    #     return depth_gt