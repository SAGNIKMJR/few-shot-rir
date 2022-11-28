import torch.nn as nn
import torch
import torch.nn.functional as F


class TransformerMemory(nn.Module):
    def __init__(self, trainer_cfg, env_cfg,):
        """
        Transformer memory
        :param trainer_cfg: trainer config
        :param env_cfg: environment config
        """
        super().__init__()

        self._transformer_memory_net_cfg = trainer_cfg.MemoryNet.Transformer
        self._env_cfg = env_cfg

        self.transformer = nn.Transformer(
            d_model=self._transformer_memory_net_cfg.input_size,
            nhead=self._transformer_memory_net_cfg.nhead,
            num_encoder_layers=self._transformer_memory_net_cfg.num_encoder_layers,
            num_decoder_layers=self._transformer_memory_net_cfg.num_decoder_layers,
            dim_feedforward=self._transformer_memory_net_cfg.hidden_size,
            dropout=self._transformer_memory_net_cfg.dropout,
            activation=self._transformer_memory_net_cfg.activation,
        )

        if trainer_cfg.encode_each_modality_as_independent_context_entry:
            context_length_multiplier = 2
        else:
            raise ValueError

        if self._env_cfg.MAX_CONTEXT_LENGTH > 0:
            self._src_mask = self._convert_attn_masks_to_transformer_format(
                torch.ones((self._env_cfg.MAX_CONTEXT_LENGTH * context_length_multiplier,
                            self._env_cfg.MAX_CONTEXT_LENGTH * context_length_multiplier,))
            )
            self._mem_mask = self._convert_attn_masks_to_transformer_format(
                torch.ones((self._env_cfg.MAX_QUERY_LENGTH, self._env_cfg.MAX_CONTEXT_LENGTH * context_length_multiplier,))
            )
        else:
            self._src_mask = self._convert_attn_masks_to_transformer_format(
                torch.zeros(((self._env_cfg.MAX_CONTEXT_LENGTH + 1) * context_length_multiplier,
                             (self._env_cfg.MAX_CONTEXT_LENGTH + 1) * context_length_multiplier,))
            )
            self._mem_mask = self._convert_attn_masks_to_transformer_format(
                torch.zeros((self._env_cfg.MAX_QUERY_LENGTH,
                             (self._env_cfg.MAX_CONTEXT_LENGTH + 1) * context_length_multiplier,))
            )

        self._tgt_mask = self._convert_attn_masks_to_transformer_format(
            torch.eye(self._env_cfg.MAX_QUERY_LENGTH)
        )

    def _convert_key_padding_masks_to_transformer_format(self, key_padding_masks):
        r"""The key_padding_masks is a FloatTensor with
            -   0 for invalid locations, and
            -   1 for valid locations.
        The required format is a BoolTensor with
            -   True for invalid locations, and
            -   False for valid locations

        source:
            - https://pytorch.org/docs/1.4.0/_modules/torch/nn/modules/transformer.html#TransformerDecoder
            - https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
        """
        return (1 - key_padding_masks) > 0

    def _convert_attn_masks_to_transformer_format(self, attn_masks):
        r"""The attn_masks is a FloatTensor with
            -   0 for invalid locations, and
            -   1 for valid locations.
        The required format is a FloatTensor with
            -   float('-inf') for invalid locations, and
            -   0. for valid locations

        source:
            - https://pytorch.org/docs/1.4.0/_modules/torch/nn/modules/transformer.html#TransformerDecoder
            - https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
        """
        return attn_masks.float().masked_fill(attn_masks == 0, float('-inf')).masked_fill(attn_masks == 1, float(0.0))

    def forward(self, observations):
        """
        does forward pass inside transformer memory
        :param observations: observations
        :return: transformer memory output features
        """
        assert "src_feats" in observations
        src_feats = observations["src_feats"]

        assert "tgt_feats" in observations
        tgt_feats = observations["tgt_feats"]

        # how masks works -- source: https://github.com/pytorch/pytorch/blob/7f73f1d591afba823daa4a99a939217fb54d7688/torch/nn/functional.py#L3360
        assert "src_key_padding_mask" in observations
        src_key_padding_mask = self._convert_key_padding_masks_to_transformer_format(observations["src_key_padding_mask"])

        assert "memory_key_padding_mask" in observations
        memory_key_padding_mask = self._convert_key_padding_masks_to_transformer_format(observations["memory_key_padding_mask"])

        self._src_mask = self._src_mask.to(src_feats.device)
        self._mem_mask = self._mem_mask.to(memory_key_padding_mask.device)
        self._tgt_mask = self._tgt_mask.to(tgt_feats.device)

        out = self.transformer(
            src_feats,
            tgt_feats,
            src_mask=self._src_mask,
            tgt_mask=self._tgt_mask,
            memory_mask=self._mem_mask,
            src_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return out
