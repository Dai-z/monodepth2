# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
from tqdm import tqdm


# Freeze bn https://discuss.pytorch.org/t/freeze-batchnorm-layer-lead-to-nan/8385
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class Trainer:

    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        # Device settings
        self.distributed = torch.cuda.device_count() > 1
        if self.distributed:
            self.device = torch.device('cuda:{}'.format(self.opt.local_rank))
            torch.cuda.set_device(self.opt.local_rank)
            torch.distributed.init_process_group(
                backend='nccl',
                init_method="env://",
            )
        else:
            self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.model = FullModel(self.opt, self.device)
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = self.model.to(self.device)
        self.model = DDP(self.model,
                         device_ids=[self.opt.local_rank],
                         output_device=self.opt.local_rank,
                         find_unused_parameters=True)

        self.model_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.local_rank == 0:
            print("Training model named:\n  ", self.opt.model_name)
            print("Models and tensorboard events files are saved to:\n  ",
                  self.opt.log_dir)
            print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {
            "kitti": datasets.KITTIRAWDataset,
            "kitti_odom": datasets.KITTIOdomDataset
        }
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits",
                             self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # Prepare datasets
        train_dataset = self.dataset(self.opt.data_path,
                                     train_filenames,
                                     self.opt.height,
                                     self.opt.width,
                                     self.opt.frame_ids,
                                     4,
                                     is_train=True,
                                     img_ext=img_ext)
        if self.distributed:
            self.train_sampler = DistributedSampler(train_dataset)
        else:
            self.train_sampler = None
        self.opt.num_workers = self.opt.num_workers // torch.cuda.device_count()
        self.train_loader = DataLoader(train_dataset,
                                       self.opt.batch_size,
                                       shuffle=(self.train_sampler is None),
                                       num_workers=self.opt.num_workers,
                                       pin_memory=True,
                                       drop_last=True,
                                       sampler=self.train_sampler)
        val_dataset = self.dataset(self.opt.data_path,
                                   val_filenames,
                                   self.opt.height,
                                   self.opt.width,
                                   self.opt.frame_ids,
                                   4,
                                   is_train=False,
                                   img_ext=img_ext)
        if self.distributed:
            self.val_sampler = DistributedSampler(train_dataset)
        else:
            self.val_sampler = None
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            shuffle=False,
            #  num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=self.val_sampler)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(
                self.log_path, mode))

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2",
            "da/a3"
        ]

        if self.opt.local_rank == 0:
            print("Using split:\n  ", self.opt.split)
            print("There are {:d} training items and {:d} validation items\n".
                  format(len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.opt.num_epochs):
            if not self.train_sampler is None:
                self.train_sampler.set_epoch(self.epoch)
            self.run_epoch()
            if (self.epoch + 1
               ) % self.opt.save_frequency == 0 and self.opt.local_rank == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        if self.opt.local_rank == 0:
            print("Training")
        self.set_train()

        if self.opt.local_rank == 0:
            t = tqdm(total=len(self.train_loader))
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.model(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if self.opt.local_rank == 0:
                if (early_phase or late_phase):
                    self.log_time(batch_idx, duration,
                                  losses["loss"].cpu().data)

                    if "depth_gt" in inputs:
                        self.compute_depth_losses(inputs, outputs, losses)

                    self.log("train", inputs, outputs, losses)
                    # FIXME: Error in val.
                    # self.val()
                t.update()

            self.step += 1
        self.model_lr_scheduler.step()

    def val(self):
        """validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.model(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(
            F.interpolate(depth_pred, [375, 1242],
                          mode="bilinear",
                          align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.cpu().detach()

        depth_gt = inputs["depth_gt"].cpu()
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred = depth_pred * torch.median(depth_gt) / torch.median(
            depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step -
                              1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(
            print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                sec_to_hm_str(time_sofar),
                                sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(
                4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image("color_{}_{}/{}".format(frame_id, s, j),
                                     inputs[("color", frame_id, s)][j].data,
                                     self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image("disp_{}/{}".format(s, j),
                                 normalize_image(outputs[("disp", s)][j]),
                                 self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j,
                                                                    f_idx][None,
                                                                           ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None,
                                                                      ...],
                        self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models",
                                   "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if self.distributed:
            self.model.module.save_model(save_folder)
        else:
            self.model.save_model(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)

        if self.distributed:
            self.model.module.load_model()
        else:
            self.model.load_model()

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(
            self.opt.load_weights_folder))

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder,
                                           "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


class FullModel(nn.Module):

    def __init__(self, opt, device):
        super(FullModel, self).__init__()
        self.opt = opt

        self.device = device
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        self.num_scales = len(self.opt.scales)

        self.use_pose_net = not (self.opt.use_stereo and
                                 self.opt.frame_ids == [0])
        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.module_names = ['encoder', 'decoder']
        # Define modules
        # Encoder
        self.encoder = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            sparse=True)
        self.encoder.to(self.device)
        num_ch_enc = self.encoder.num_ch_enc

        # Depth decoder
        self.depth = networks.DepthDecoder(num_ch_enc, self.opt.scales)
        self.depth.to(self.device)

        # Pose net
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                # Pose encoder
                self.pose_encoder = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.pose_encoder.to(self.device)
                num_ch_pose_enc = self.pose_encoder.num_ch_enc
                self.module_names.append('pose_encoder')
                # pose decoder
                self.pose = networks.PoseDecoder(num_ch_pose_enc,
                                                 num_input_features=1,
                                                 num_frames_to_predict_for=2)
            elif self.opt.pose_model_type == "shared":
                self.pose = networks.PoseDecoder(num_ch_enc,
                                                 self.num_pose_frames)
            elif self.opt.pose_model_type == "posecnn":
                self.pose = networks.PoseCNN(self.num_input_frames if self.opt.
                                             pose_model_input == "all" else 2)
            self.pose.to(self.device)
            self.module_names.append('pose')

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.predictive_mask = networks.DepthDecoder(
                num_ch_enc,
                self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.predictive_mask.to(self.device)
            self.module_names.append('predictive_mask')

        # define functions
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2**scale)
            w = self.opt.width // (2**scale)

            self.backproject_depth[scale] = BackprojectDepth(
                self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

    def forward(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.encoder(all_color_aug)
            all_features = [
                torch.split(f, self.opt.batch_size) for f in all_features
            ]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.depth(features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.encoder(
                torch.cat((inputs["color_aug", 0, 0], inputs['lidar', 0, 0]),
                          1))
            outputs = self.depth(features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.predictive_mask(features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        """
        Distribute the loss on multi-gpu to reduce 
        the memory cost in the main gpu.
        You can check the following discussion.
        https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
        """

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # in this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {
                    f_i: inputs["color_aug", f_i, 0]
                    for f_i in self.opt.frame_ids
                }

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # to maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        tmp_inputs = torch.cat(pose_inputs, 1)
                        tmp_inputs = self.pose_encoder(tmp_inputs)
                        pose_inputs = [tmp_inputs]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.pose(pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # invert the matrix if the frame id is negative
                    outputs[("cam_t_cam", 0,
                             f_i)] = transformation_from_parameters(
                                 axisangle[:, 0],
                                 translation[:, 0],
                                 invert=(f_i < 0))
        else:
            # here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat([
                    inputs[("color_aug", i, 0)]
                    for i in self.opt.frame_ids
                    if i != "s"
                ], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.pose_encoder(pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [
                    features[i] for i in self.opt.frame_ids if i != "s"
                ]

            axisangle, translation = self.pose(pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_t_cam", 0,
                             f_i)] = transformation_from_parameters(
                                 axisangle[:, i], translation[:, i])

        return outputs

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            sparse_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                lidar = inputs[('lidar', frame_id, 0)]

                lidar_mask = lidar > 0
                sparse_losses.append(
                    self.compute_sparse_loss(pred, lidar, lidar_mask))
                # FIXME: Add mask
                reprojection_losses.append(
                    self.compute_reprojection_loss(pred,
                                                   target))  #, (lidar <= 0)))

            reprojection_losses = torch.cat(reprojection_losses, 1)
            sparse_losses = torch.cat(sparse_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(
                    identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(
                        1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(mask,
                                         [self.opt.height, self.opt.width],
                                         mode="bilinear",
                                         align_corners=False)

                reprojection_losses = mask * reprojection_losses

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(
                    mask.shape).cuda())
                loss = loss + weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss = identity_reprojection_loss + torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss,
                                      reprojection_loss, sparse_losses),
                                     dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            # reprojection & saprse loss
            loss = loss + to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            # smooth loss
            loss = loss + self.opt.disparity_smoothness * smooth_loss / (2**
                                                                         scale)
            total_loss = total_loss + loss
            losses["loss/{}".format(scale)] = loss

        total_loss = total_loss / self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_sparse_loss(self, pred, lidar, mask):
        """Computes sparse loss between a batch of predicted and lidar points
        """
        selected_pred = mask * pred
        selected_lidar = mask * lidar
        abs_diff = torch.abs(selected_lidar - selected_pred)
        l1_loss = abs_diff.mean(1, True)
        return l1_loss

    def compute_reprojection_loss(self, pred, target, mask=None):
        """Computes reprojection loss between a batch of predicted and target images
        """
        if not mask is None:
            selected_target = mask * target
            selected_pred = mask * pred
            abs_diff = torch.abs(selected_target - selected_pred)
        else:
            abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def generate_images_pred(self, inputs, outputs):
        """generate the warped (reprojected) color images for a minibatch.
        generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width],
                                     mode="bilinear",
                                     align_corners=False)
                source_scale = 0

            # _, depth = disp_to_depth(disp, self.opt.min_depth,
            #                          self.opt.max_depth)
            # Directly predict depth result
            depth = disp
            outputs[("disp", 0, scale)] = 1 / (depth + 1e-7)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    t = inputs["stereo_t"]
                else:
                    T = outputs[("cam_t_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    t = transformation_from_parameters(
                        axisangle[:, 0],
                        translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def save_model(self, save_folder):
        """Save model weights to disk
        """

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

    def load_model(self):
        """Load model(s) from disk
        """

        for name in self.opt.models_to_load:
            print("Loading {} weights...".format(name))
            path = os.path.join(self.opt.load_weights_folder,
                                "{}.pth".format(name))
            m = getattr(self, name)
            model_dict = m.state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            m.load_state_dict(model_dict)
