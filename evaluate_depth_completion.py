from __future__ import absolute_import, division, print_function

import os
from cv2 import cv2
import numpy as np

import torch
from torch.nn import functional as F
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


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def optimize_with_lidar(opt, encoder, depth_decoder, data):
    params = []
    for _, param in encoder.named_parameters():
        if param.requires_grad:
            params.append(param)
    for _, param in depth_decoder.named_parameters():
        if param.requires_grad:
            params.append(param)
    optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
    prev_loss = 0

    input_color = data[("color", 0, 0)].cuda()
    input_lidar = data[("lidar", 0, 0)].cuda()
    if opt.sparse:
        input_data = torch.cat((input_color, input_lidar), 1)
    else:
        input_data = input_color
    for idx_iter in range(opt.num_iters):
        output = depth_decoder(encoder(input_data))
        loss = 0
        # if idx_iter <= 10:
        #     set_lr(optimizer, opt.learning_rate * (idx_iter + 1) / 10)
        if -1 in opt.scales:
            height, width = data[("lidar", 0, -1)].shape[2:]
            output['disp', -1] = F.interpolate(output['disp', 0],
                                               [height, width],
                                               mode="bilinear",
                                               align_corners=False)
        for scale in opt.scales:
            lidar = data[("lidar", 0, scale)].cuda() * 80
            disp = output[("disp", scale)]
            mask = lidar > 0
            _, pred = disp_to_depth(disp, opt.min_depth, opt.max_depth)
            selected_pred = pred[mask]
            selected_lidar = lidar[mask]
            if opt.median_scaling:
                ratio = float((selected_lidar / selected_pred).median())
                selected_pred *= ratio
            loss += ((selected_lidar - selected_pred)**2).mean()
        # print(loss)
        # if abs(prev_loss - loss) / loss < 1e-2:
        #     break
        prev_loss = float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return depth_decoder(encoder(input_data))


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
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path,
                        fix_imports=True,
                        encoding='latin1',
                        allow_pickle=True)["data"]
    if opt.ext_disp_to_eval is None:
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        print(opt.eval_split)
        dataset = datasets.KITTIDepthDataset(opt.data_path,
                                             filenames,
                                             encoder_dict['height'],
                                             encoder_dict['width'], [0],
                                             4,
                                             is_train=False,
                                             img_ext='.png')
        dataloader = DataLoader(dataset,
                                opt.batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers,
                                pin_memory=True,
                                drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers,
                                         False,
                                         sparse=opt.sparse)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc,
                                              sparse=opt.sparse)

        model_dict = encoder.state_dict()

        encoder_state = {
            k.replace('module.', ''): v
            for k, v in encoder_dict.items()
            if k.replace('module.', '') in model_dict
        }
        encoder.load_state_dict(encoder_state)
        decoder_state = torch.load(decoder_path)
        decoder_state = {
            k.replace('module.', ''): v for k, v in decoder_state.items()
        }
        depth_decoder.load_state_dict(decoder_state)

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        model_info = 'with' if opt.sparse else 'without'
        print("-> Model {} lidar input.".format(model_info))

        if opt.batch_size > 1:
            iterator = tqdm(dataloader)
        else:
            iterator = dataloader
        for idx, data in enumerate(iterator):
            if opt.optimize:
                encoder.load_state_dict(encoder_state)
                depth_decoder.load_state_dict(decoder_state)
                output = optimize_with_lidar(opt, encoder, depth_decoder, data)
            else:
                input_color = data[("color", 0, 0)].cuda()
                input_lidar = data[("lidar", 0, 0)].cuda()
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat(
                        (input_color, torch.flip(input_color, [3])), 0)
                if opt.sparse:
                    input_data = torch.cat((input_color, input_lidar), 1)
                else:
                    input_data = input_color
                output = depth_decoder(encoder(input_data))

            if opt.sparse:
                # pred_disp = output[("disp", 0)].cpu()[:, 0].numpy()
                pred_disp, pred_depth = disp_to_depth(output[("disp", 0)],
                                                      opt.min_depth,
                                                      opt.max_depth)
            else:
                pred_disp, pred_depth = disp_to_depth(output[("disp", 0)],
                                                      opt.min_depth,
                                                      opt.max_depth)
            pred_disp = pred_disp.detach().cpu()[:, 0].numpy()
            pred_depth = pred_depth.detach().cpu()[0, 0].numpy()

            if opt.batch_size == 1:
                if opt.median_scaling:
                    lidar = data[("lidar", 0, 0)][0, 0] * 80
                    mask = lidar > 0
                    ratio = float((lidar[mask] / pred_depth[mask]).median())
                    pred_depth *= ratio
                # gt_depth = gt_depths[idx]
                gt_depth = data["depth_gt"][0, 0].numpy()
                gt_height, gt_width = gt_depth.shape[:2]
                pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
                pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
                pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
                mask = gt_depth > 0
                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]
                print('[{}]'.format(idx), compute_errors(gt_depth, pred_depth))

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(
                    pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)

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

    print("   Mono evaluation - {} using median scaling".format(
        "" if opt.median_scaling else "not"))

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
        if opt.sparse:
            # pred_depth = pred_disp
            pred_depth = 1 / pred_disp
        else:
            pred_depth = 1 / pred_disp
        mask = gt_depth > 0

        pred_depth *= opt.pred_depth_scale_factor
        # Calc ratios
        ratio_image = velodyne_depth[velodyne_mask] / pred_depth[velodyne_mask]
        ratio = ratio_image.mean()
        stds.append(np.std(ratio_image))

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
