# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 num_input_images=1,
                 sparse=False,
                 skip_bn=False):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        if sparse:
            self.conv_i = nn.Conv2d(3 * num_input_images,
                                    48,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    bias=False)
            self.bn_i = nn.BatchNorm2d(48)
            self.conv_s = nn.Conv2d(3 * num_input_images,
                                    16,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    bias=False)
            self.bn_s = nn.BatchNorm2d(16)
        else:
            self.conv1 = nn.Conv2d(3 * num_input_images,
                                   64,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], skip_bn=skip_bn)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       skip_bn=skip_bn)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       skip_bn=skip_bn)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       skip_bn=skip_bn)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers,
                            pretrained=False,
                            num_input_images=1,
                            sparse=False,
                            skip_bn=False):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {
        18: models.resnet.BasicBlock,
        50: models.resnet.Bottleneck
    }[num_layers]
    model = ResNetMultiImageInput(block_type,
                                  blocks,
                                  num_input_images=num_input_images,
                                  sparse=sparse,
                                  skip_bn=skip_bn)

    if pretrained:
        loaded = model_zoo.load_url(
            models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded, strict=False)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self,
                 num_layers,
                 pretrained,
                 num_input_images=1,
                 sparse=False,
                 skip_bn=False):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152
        }
        self.sparse = sparse

        if num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers))

        self.num_input_images = num_input_images
        if num_input_images > 1 or sparse:
            self.encoder = resnet_multiimage_input(num_layers, pretrained,
                                                   num_input_images, sparse,
                                                   skip_bn)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        self.skip_bn = skip_bn

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        if self.sparse:
            feat_i = self.encoder.bn_i(
                self.encoder.conv_i(x[:, 3 * self.num_input_images:, :, :]))
            # feat_i = self.encoder.bn_i(feat_i)
            feat_s = self.encoder.bn_s(
                self.encoder.conv_s(x[:, 3 * self.num_input_images:, :, :]))
            # feat_s = self.encoder.bn_s(feat_s)
            # x = torch.cat((feat_i, feat_s), 1)
            self.features.append(torch.cat((feat_i, feat_s), 1))
        else:
            out = self.encoder.conv1(x)
            if not self.skip_bn:
                out = self.encoder.bn1(out)
            self.features.append(self.encoder.relu(out))
        self.features.append(
            self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
