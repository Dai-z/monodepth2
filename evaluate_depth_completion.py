from __future__ import absolute_import, division, print_function

import os
from cv2 import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines, printc
from options import MonodepthOptions
import datasets
import networks

from tqdm import tqdm
import PIL.Image as pil
import time

cv2.setNumThreads(
    0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred)**2
    # [m] to [mm]
    rmse = np.sqrt(rmse[gt.nonzero()].mean()) * 1000
    mae = np.abs(gt - pred)[gt.nonzero()].mean() * 1000

    return rmse, mae, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    opt.eval_split = 'completion'
    opt.eval_mono = 1
    opt.eval_stereo = 0

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    filenames = readlines(
        os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    if opt.ext_disp_to_eval is None:
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        print(opt.eval_split)
        dataset = datasets.KITTIRAWDataset(opt.data_path,
                                           filenames,
                                           encoder_dict['height'],
                                           encoder_dict['width'], [0],
                                           4,
                                           is_train=False,
                                           img_ext='.png')
        dataloader = DataLoader(dataset,
                                16,
                                shuffle=False,
                                num_workers=opt.num_workers,
                                pin_memory=True,
                                drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers,
                                         False,
                                         sparse=not opt.no_sparse)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()

        encoder.load_state_dict({
            k.replace('module.', ''): v
            for k, v in encoder_dict.items()
            if k.replace('module.', '') in model_dict
        })

        decoder_dict = torch.load(decoder_path)
        corrected_dict = {
            k.replace('module.', ''): v for k, v in decoder_dict.items()
        }
        depth_decoder.load_state_dict(corrected_dict)

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        model_info = 'without' if opt.no_sparse else 'with'
        print("-> Model {} lidar input.".format(model_info))

        t = tqdm(total=len(dataloader))
        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
                input_lidar = data[("lidar", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat(
                        (input_color, torch.flip(input_color, [3])), 0)
                if opt.no_sparse:
                    input_data = input_color
                else:
                    input_data = torch.cat((input_color, input_lidar), 1)

                output = depth_decoder(encoder(input_data))

                pred_disp = output[("disp", 0)].cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(
                        pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                t.update()

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark",
                             "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.ext_disp_to_eval is None and opt.save_pred_disps:
        output_path = os.path.join(opt.load_weights_folder,
                                   "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.ext_disp_to_eval is None and opt.eval_split in [
            'benchmark', 'completion'
    ]:
        save_dir = os.path.join(opt.load_weights_folder,
                                "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        # print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        # quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path,
                        fix_imports=True,
                        encoding='latin1',
                        allow_pickle=True)["data"]

    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    stds = []

    side_map = {"2": '2', "3": '3', "l": '2', "r": '3'}
    for i in range(pred_disps.shape[0]):

        # Get projected velodyne depth
        folder, frame_id, side = filenames[i].split()
        frame_id = int(frame_id)
        velodyne_path = os.path.join(opt.data_path, folder, "proj_depth",
                                     "velodyne_raw", "image_0" + side_map[side],
                                     "{:010d}.png".format(frame_id))

        velodyne_depth = cv2.imread(velodyne_path, cv2.IMREAD_ANYDEPTH).astype(
            np.float32) / 256
        velodyne_mask = velodyne_depth > 1e-7
        mask_idx = velodyne_mask.nonzero()

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        # FIXME: using depth as disp now
        # pred_depth = 1 / pred_disp
        pred_depth = pred_disp
        mask = gt_depth > 0

        pred_depth *= opt.pred_depth_scale_factor
        # Calc ratios
        # ratio = np.median(gt_depth) / np.median(pred_depth)
        # ratio = np.mean(gt_depth) / np.mean(pred_depth)
        ratio = np.mean(velodyne_depth[velodyne_mask] /
                        pred_depth[velodyne_mask])
        ratio_image = velodyne_depth[velodyne_mask] / pred_depth[velodyne_mask]
        ratio = ratio_image.mean()
        # print('ratio std {}: {}'.format(ratio_image.mean(), np.std(ratio_image)))
        stds.append(np.std(ratio_image))

        # vis_var = np.zeros(pred_depth.shape)
        # var_image = np.log(abs(ratio_image - ratio))
        # vis_var[mask_idx] = var_image
        # max_var = vis_var.max()
        # # Create color bar
        # colorbar = np.uint8(
        #     np.repeat(np.linspace(255, 0,
        #                           num=pred_depth.shape[0])[:, np.newaxis],
        #               100,
        #               axis=1)[:, :, np.newaxis])
        # colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_JET)
        # colorbar = cv2.putText(colorbar, '{:.2E}'.format(vis_var.max()),
        #                        (0, 10), cv2.FONT_HERSHEY_COMPLEX, .5,
        #                        (204, 255, 102), 1)
        # # visualize variance
        # vis_var /= max_var
        # vis_var = np.uint8(vis_var * 255)
        # vis_var = cv2.applyColorMap(vis_var, cv2.COLORMAP_JET)
        # vis_var[:,:,0] *= velodyne_mask
        # vis_var[:,:,1] *= velodyne_mask
        # vis_var[:,:,2] *= velodyne_mask
        # vis_var = np.concatenate((vis_var, colorbar), axis=1)
        # cv2.imwrite('log_var.jpg', vis_var)
        # time.sleep(.1)
        # exit()

        ratios.append(ratio)
        if opt.median_scaling:
            pred_depth *= ratio

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    stds = np.array(stds)
    print('std in single image | max: {:.3f}, min: {:.3f}'.format(
        np.max(stds), np.min(stds)))

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(
        " Scaling ratios | min: {:.3f} | max: {:.3f} | med: {:0.3f} | std: {:0.3f}"
        .format(ratios.min(), ratios.max(), med, np.std(ratios)))

    mean_errors = np.array(errors).mean(0)

    print("\n| " + ("{:>8} | " * 5).format("rmse", "mae", "a1", "a2", "a3"))
    print(("| {:8.3f} " * 5).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
