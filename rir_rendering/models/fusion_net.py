import numpy as np
import torch
import torch.nn as nn

from rir_rendering.common.utils import Flatten


class FusionNet(nn.Module):
    def __init__(self, trainer_cfg, n_input_feats,):
        """1 linear fusion net.

        Takes in 1D observation encodings and produces src embeddings for transformer encoder
        (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer.forward)

        :param  trainer_cfg: The size of the embedding vector
        :param n_input_feats: number of input features
        """
        super().__init__()

        self._trainer_cfg = trainer_cfg
        self._memory_net_cfg = trainer_cfg.MemoryNet
        self._fusion_enc_cfg = trainer_cfg.FusionEnc

        self._n_input_feats = n_input_feats
        self._input_size = None

        self.net = nn.Sequential()

        if self._memory_net_cfg.type == "transformer":
            self._transformer_memory_net_cfg = self._memory_net_cfg.Transformer
            self._input_size = self._transformer_memory_net_cfg.input_size
            # using default kaimin-uniform init, source: https://pytorch.org/docs/1.4.0/_modules/torch/nn/modules/linear.html#Linear
            if n_input_feats != self._input_size:
                self.net = nn.Sequential(
                    nn.Linear(n_input_feats, self._input_size, bias=False),
                )
        else:
            raise ValueError

    def layer_init(self):
        """
        initalizes the layer parameters
        :return: None
        """
        for layer in self.net:
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

    def forward(self, observations):
        """
        does forward pass in fusion layer
        :param observations: observations
        :return: fused features
        """
        net_input = []
        for observation in observations:
            assert len(observation.size()) == 2

            net_input.append(observation)

        if self._fusion_enc_cfg.type == "concatenate":
            net_input = torch.cat(net_input, dim=-1)
            assert len(net_input.size()) == 2
        else:
            raise NotImplementedError

        if net_input.size(-1) != self._input_size:
            assert net_input.size(-1) == self._n_input_feats

        return self.net(net_input)
