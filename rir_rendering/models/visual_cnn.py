import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from rir_rendering.common.utils import Flatten


class VisualEnc(nn.Module):
    def __init__(self, ):
        """
        ResNet-18.
        Takes in observations and produces an embedding of the rgb and depth components
        """
        super().__init__()

        self._n_input_rgb = 3
        self._n_input_depth = 1

        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc_backup = self.cnn.fc
        self.cnn.fc = nn.Sequential()

        if self._n_input_rgb + self._n_input_depth != 3:
            self.cnn.conv1 = nn.Conv2d(self._n_input_rgb + self._n_input_depth,
                                       self.cnn.conv1.out_channels,
                                       kernel_size=self.cnn.conv1.kernel_size,
                                       stride=self.cnn.conv1.stride,
                                       padding=self.cnn.conv1.padding,
                                       bias=False)

            nn.init.kaiming_normal_(
                self.cnn.conv1.weight, mode="fan_out", nonlinearity="relu",
            )

    @property
    def is_blind(self):
        """
        get if network produces any output features or not
        :return: if network produces any output features or not
        """
        return False

    @property
    def n_out_feats(self):
        """
        get number of visual encoder output features
        :return: number of visual encoder output features
        """
        if self.is_blind:
            return 0
        else:
            # resnet-18
            return 512

    def forward(self, observations,):
        """
        does forward pass in visual encoder
        :param observations: observations
        :return: visual features
        """
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations.float() / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.float().permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)
