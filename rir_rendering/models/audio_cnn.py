import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from rir_rendering.common.utils import Flatten


def unet_upconv(input_nc, output_nc, kernel_size=(4, 4), outermost=False, norm_layer=nn.BatchNorm2d, stride=(2, 2),
                padding=(1, 1), output_padding=(0, 0), bias=False,):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                                output_padding=output_padding, bias=bias)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])


class AudioEnc(nn.Module):
    def __init__(self, audio_cfg,  log_instead_of_log1p_in_logspace=False,
                 log_eps=1.0e-8, ):
        """
        ResNet-18.
        Takes in observations (binaural IR magnitude spectrograms) and produces an acoustic embedding

        :param audio_cfg: audio config
        :param log_instead_of_log1p_in_logspace: compute log of magnitude spect. instead of log(1 + ...)
        :param log_eps: epsilon to be used to compute log for numerical stability
        """
        super().__init__()

        self._audio_cfg = audio_cfg
        self._log_instead_of_log1p_in_logspace = log_instead_of_log1p_in_logspace
        self._log_eps = log_eps

        self._n_input = 2

        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc_backup = self.cnn.fc
        self.cnn.fc = nn.Sequential()

        self.cnn.conv1 = nn.Conv2d(self._n_input,
                                   self.cnn.conv1.out_channels,
                                   kernel_size=self.cnn.conv1.kernel_size,
                                   stride=self.cnn.conv1.stride,
                                   padding=self.cnn.conv1.padding,
                                   bias=False)

        nn.init.kaiming_normal_(
            self.cnn.conv1.weight, mode="fan_out", nonlinearity="relu",
        )

    @property
    def n_out_feats(self):
        """
        get number of audio encoder features
        :return: number of audio encoder features
        """
        # resnet-18
        return 512

    def forward(self, observations,):
        """
        does forward pass in audio  encoder
        :param observations: observations
        :return: acoustic/audio features
        """
        cnn_input = []

        assert torch.all(observations["audio_spect"][..., :2] >= 0).item(), "first 2 channels should have magnitude"

        if self._log_instead_of_log1p_in_logspace:
            audio_spect_observations = torch.log(observations["audio_spect"] + self._log_eps)
        else:
            audio_spect_observations = torch.log1p(observations["audio_spect"])

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        audio_spect_observations = audio_spect_observations.permute(0, 3, 1, 2)
        cnn_input.append(audio_spect_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)


class AudioDec(nn.Module):
    def __init__(self, trainer_cfg, audio_cfg,):
        """
        a stack of U-Net upconvs
        Takes in transformer feature outputs and produces estimates of IR magnitude spectrograms

        :param trainer_cfg: trainer config
        :param audio_cfg: audio config
        """
        super().__init__()

        self._trainer_cfg = trainer_cfg
        self._memory_net_cfg = trainer_cfg.MemoryNet

        self._audio_cfg = audio_cfg

        assert self._memory_net_cfg.type == "transformer"
        self._input_size = self._memory_net_cfg.Transformer.input_size
        assert self._input_size == 1024

        self._n_out = 2

        self._n_input_channels = 64
        self._n_input_h = 4
        self._n_input_w = 4

        self.cnn =\
            nn.Sequential(
                unet_upconv(64 , 64 * 8),
                unet_upconv(64 * 8, 64 * 4,),
                unet_upconv(64 * 4, 64 * 2,),
                unet_upconv(64 * 2, 64 * 1,),
                unet_upconv(64, 32,),
                unet_upconv(32, 16, output_padding=(0, 1)),
                nn.Sequential(nn.Conv2d(16, self._n_out, kernel_size=(3, 3), padding=(1, 2), stride=(1, 1), bias=False)),
            )

        self.layer_init()

    def layer_init(self):
        """
        initalizes the layer parameters
        :return: None
        """
        for module in self.cnn:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("relu")
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if layer.affine:
                        layer.weight.data.fill_(1)
                        layer.bias.data.zero_()

    def forward(self, observations,):
        """
        does forward pass in audio decoder
        :param observations: observations
        :return: estimated IR magnitude spectrograms
        """
        cnn_input = []

        assert "memory_out_feats" in observations
        memory_out_feats = observations["memory_out_feats"]
        assert len(memory_out_feats.size()) == 2
        assert memory_out_feats.size(1) == self._input_size

        assert self._n_input_channels is not None
        assert self._n_input_h is not None
        assert self._n_input_w is not None
        memory_out_feats =\
            memory_out_feats.reshape((memory_out_feats.size(0),
                                      self._n_input_channels,
                                      self._n_input_h,
                                      self._n_input_w))
        cnn_input.append(memory_out_feats)
        cnn_input = torch.cat(cnn_input, dim=1)

        out = self.cnn(cnn_input)
        assert len(out.size()) == 4
        # permute tensor to dimension [BATCH x HEIGHT x WIDTH x CHANNEL]
        out = out.permute(0, 2, 3, 1)

        return out
