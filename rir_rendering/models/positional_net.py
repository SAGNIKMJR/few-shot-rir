import numpy as np
import torch
import torch.nn as nn

from rir_rendering.common.utils import Flatten


# min freq for sinusoidal positional encodings, source: https://arxiv.org/pdf/1706.03762.pdf
MIN_FREQ = 1e-4


class PositionalEnc(nn.Module):
    def __init__(self, positional_enc_cfg):
        """
        Takes in positional attributes and produces and produces their embeddings
        :param positional_enc_cfg: positional encoder config
        """
        super().__init__()

        self._n_positional_obs = 5

        self._positional_enc_cfg = positional_enc_cfg
        self._freqs = None
        self._pi = torch.acos(torch.zeros(1)).item() * 2

        if self._positional_enc_cfg.type == "sinusoidal":
            # source: 1. https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb
            #         2. https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
            self._freqs = MIN_FREQ ** (2 * (torch.arange(self._positional_enc_cfg.num_freqs_for_sinusoidal,
                                                         dtype=torch.float32) // 2) /
                                       self._positional_enc_cfg.num_freqs_for_sinusoidal)
        else:
            raise ValueError

    @property
    def n_out_feats(self):
        """
        get number of pose encoder output features
        :return: number of pose encoder output features
        """
        if self._positional_enc_cfg.type in ["sinusoidal"]:
            return self._freqs.size(0) * self._n_positional_obs
        else:
            raise ValueError

    def forward(self, observations):
        """
        does forward pass in pose encoder
        :param observations: observations
        :return: pose features
        """
        positional_obs = observations["positional_obs"]
        positional_enc_out = []

        if self._positional_enc_cfg.type in ["sinusoidal"]:
            assert len(positional_obs.size()) == 2
            assert positional_obs.size(-1) == self._n_positional_obs

            freqs = self._freqs.unsqueeze(0).repeat((positional_obs.size(0), 1)).to(positional_obs.device)

            for positional_obs_idx in range(self._n_positional_obs):
                positional_obs_this_idx = positional_obs[:, positional_obs_idx].unsqueeze(-1)
                positional_obs_this_idx = positional_obs_this_idx * freqs
                positional_obs_this_idx_clone = positional_obs_this_idx.clone()
                if self._positional_enc_cfg.type == "sinusoidal":
                    positional_obs_this_idx_clone[..., ::2] = torch.cos(positional_obs_this_idx[..., ::2])
                    positional_obs_this_idx_clone[..., 1::2] = torch.sin(positional_obs_this_idx[..., 1::2])
                else:
                    raise ValueError
                positional_enc_out.append(positional_obs_this_idx_clone)

            positional_enc_out = torch.cat(positional_enc_out, dim=-1)

            assert len(positional_enc_out.size()) == 2
            assert positional_enc_out.size(0) == positional_obs.size(0)
            assert positional_enc_out.size(1) == (self._freqs.size(0) * self._n_positional_obs)
        else:
            raise ValueError

        return positional_enc_out
