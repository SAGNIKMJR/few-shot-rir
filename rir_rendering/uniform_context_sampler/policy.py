import os
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

from rir_rendering.models.visual_cnn import VisualEnc
from rir_rendering.models.audio_cnn import AudioEnc, AudioDec
from rir_rendering.models.positional_net import PositionalEnc
from rir_rendering.models.fusion_net import FusionNet
from rir_rendering.models.memory_net import TransformerMemory


class Policy(nn.Module):
    def __init__(self,
                 visual_context_enc,
                 audio_context_enc,
                 modality_tag_type_lookup_dict,
                 pose_context_enc,
                 fusion_context_enc,
                 memory_net,
                 pose_query_enc,
                 fusion_query_enc,
                 audio_dec,
                 cfg,
                 ):
        """
        Network for IR prediction for uniform sampling of context
        :param visual_context_enc: visual context encoder
        :param audio_context_enc: audio (IR) context encoder
        :param modality_tag_type_lookup_dict:  modality type lookup table
        :param pose_context_enc: pose context encoder
        :param fusion_context_enc: fusion layer for context
        :param memory_net: memory network
        :param pose_query_enc: query pose encoder
        :param fusion_query_enc: fusion layer for query
        :param audio_dec: audio (IR) decoder
        :param cfg: config
        """
        super().__init__()
        self.visual_context_enc = visual_context_enc
        self.audio_context_enc = audio_context_enc
        self.modality_tag_type_lookup_dict = modality_tag_type_lookup_dict
        self.pose_context_enc = pose_context_enc
        self.fusion_context_enc = fusion_context_enc
        self.memory_net = memory_net
        self.pose_query_enc = pose_query_enc
        self.fusion_query_enc = fusion_query_enc
        self.audio_dec = audio_dec

        self._cfg = cfg

        self._task_cfg = cfg.TASK_CONFIG
        self._env_cfg = self._task_cfg.ENVIRONMENT

        self._uniform_context_sampler_cfg = cfg.UniformContextSampler

        # max_context_length: transformer source sequence / max context length (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer),
        # max_query_length: transformer target sequence / max query length (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer),
        self.max_context_length = self._env_cfg.MAX_CONTEXT_LENGTH
        if self.max_context_length == 0:
            self.max_context_length = 1
        self.max_query_length = self._env_cfg.MAX_QUERY_LENGTH

    def forward(self, observations,):
        """
        Does forward pass in IR prediction network
        :param observations: observations
        :return: estimated IR
        """
        # --------------------------------------------- context encoding ----------------------------------------------------
        context_feats = []
        if self._uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry:
            num_n_input_feats_fusion_context_enc = 2
            for idx_n_input_feats_fusion_context_enc in range(num_n_input_feats_fusion_context_enc):
                context_feats.append([])
        else:
            raise ValueError

        assert "context_views" in observations
        context_views = observations["context_views"]
        B = context_views.size(0)

        # B x max_context_length x ... -> (B * max_context_length) x ...; B: batch size,
        # max_context_length: transformer source sequence length S (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        context_views = context_views.reshape((-1, *context_views.size()[2:]))

        if self._cfg.SENSORS in [["RGB_SENSOR", "DEPTH_SENSOR"], ["DEPTH_SENSOR", "RGB_SENSOR"]]:
            visual_context_feats = self.visual_context_enc({"rgb": context_views[..., :3],
                                                            "depth": context_views[..., 3].unsqueeze(-1)})
        else:
            raise ValueError

        if self._uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry:
            context_feats[0].append(visual_context_feats)
        else:
            raise ValueError

        assert "context_echoes" in observations
        context_echoes = observations["context_echoes"]

        # B x max_context_length x ... -> (B * max_context_length) x ...; B: batch size,
        # max_context_length: transformer source sequence length S (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        context_echoes = context_echoes.reshape((-1, *context_echoes.size()[2:]))
        audio_context_feats = self.audio_context_enc({"audio_spect": context_echoes})
        if self._uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry:
            context_feats[-1].append(audio_context_feats)
        else:
            raise ValueError

        assert "context_poses" in observations
        context_poses = observations["context_poses"]

        # B x max_context_length x ... -> (B * max_context_length) x ...; B: batch size,
        # max_context_length: transformer source sequence length S (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        context_poses = context_poses.reshape((-1, *context_poses.size()[2:]))
        pose_context_feats = self.pose_context_enc({"positional_obs": context_poses})
        if self._uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry:
            for idx_context_feats in range(len(context_feats)):
                context_feats[idx_context_feats].append(pose_context_feats)
                if self._uniform_context_sampler_cfg.append_modality_type_tag_encoding_to_each_modality_encoding:
                    modality_type_tag_inp =\
                        torch.LongTensor([idx_context_feats] * pose_context_feats.size(0)).to(pose_context_feats.device)
                    modality_type_tag_encoding = self.modality_tag_type_lookup_dict(modality_type_tag_inp)
                    context_feats[idx_context_feats].append(modality_type_tag_encoding)
                else:
                    raise ValueError
        else:
            raise ValueError

        # --------------------------------------------- context fusion ----------------------------------------------------
        if self._uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry:
            fused_context_feats = []
            for idx_context_feats in range(len(context_feats)):
                assert idx_context_feats < len(self.fusion_context_enc)
                temp_fused_context_feats = self.fusion_context_enc[idx_context_feats](context_feats[idx_context_feats])
                temp_fused_context_feats = temp_fused_context_feats.reshape((B, self.max_context_length, -1))
                temp_fused_context_feats = temp_fused_context_feats.permute(1, 0, 2)
                fused_context_feats.append(temp_fused_context_feats)

            fused_context_feats = torch.cat(fused_context_feats, dim=0)
        else:
            raise ValueError

        # --------------------------------------------- context and memory key padding masks ----------------------------------------------------
        assert "context_mask" in observations
        context_key_padding_mask = observations["context_mask"]
        assert len(context_key_padding_mask.size()) == 2

        if self._uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry:
            assert num_n_input_feats_fusion_context_enc is not None
            context_key_padding_mask_lst = []
            for idx_n_input_feats_fusion_context_enc in range(num_n_input_feats_fusion_context_enc):
                context_key_padding_mask_lst.append(context_key_padding_mask)
            context_key_padding_mask = torch.cat(context_key_padding_mask_lst, dim=-1)
        else:
            raise ValueError

        memory_key_padding_mask = context_key_padding_mask.clone()

        # --------------------------------------------- query encoding ----------------------------------------------------
        assert "query_poses" in observations
        query_poses = observations["query_poses"]

        # B x max_query_length x ... -> (B * max_query_length) x ...; B: batch size,
        # max_query_length: transformer target sequence length T (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        query_poses = query_poses.reshape((-1, *query_poses.size()[2:]))
        pose_query_feats = self.pose_context_enc({"positional_obs": query_poses})

        query_feats = [pose_query_feats]

        # --------------------------------------------- query fusion ----------------------------------------------------
        fused_query_feats = self.fusion_query_enc(query_feats)

        #  (B * max_query_length) x ... -> B x max_query_length x ...; B: batch size,
        # max_query_length: transformer target sequence length T (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        fused_query_feats = fused_query_feats.reshape((B, self.max_query_length, -1))
        # B x max_query_length x ... -> max_query_length x B x -1; B: batch size,
        # max_query_length: transformer target sequence length T (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        fused_query_feats = fused_query_feats.permute(1, 0, 2)

        # --------------------------------------------- query key padding mask ----------------------------------------------------
        assert "query_mask" in observations
        query_key_padding_mask = observations["query_mask"]
        assert len(query_key_padding_mask.size()) == 2

        # --------------------------------------------- self-attention memory  ----------------------------------------------------
        memory_out_feats =\
            self.memory_net(
                {
                    "src_feats": fused_context_feats,
                    "tgt_feats": fused_query_feats,
                    "src_key_padding_mask": context_key_padding_mask,
                    "tgt_key_padding_mask": query_key_padding_mask,
                    "memory_key_padding_mask": memory_key_padding_mask,
                }
            )

        # max_query_length x B x ... -> B x max_query_length x ...; B: batch size,
        # max_query_length: transformer target sequence length T (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        memory_out_feats = memory_out_feats.permute(1, 0, 2)
        # B x max_query_length x ... -> (B * max_query_length) x ...; B: batch size,
        # max_query_length: transformer target sequence length T (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        memory_out_feats = memory_out_feats.reshape((-1, *memory_out_feats.size()[2:]))

        # --------------------------------------------- audio decoding ----------------------------------------------------
        pred_queryIR = self.audio_dec({"memory_out_feats": memory_out_feats})

        # (B * max_query_length) x ... -> B x max_query_length x ...; B: batch size,
        # max_query_length: transformer target sequence length T (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        pred_queryIR = pred_queryIR.reshape((B,
                                             self.max_query_length,
                                             *pred_queryIR.size()[1:]))

        return pred_queryIR


class UniformContextSamplerPolicy(Policy):
    def __init__(
        self,
        cfg,
    ):
        """
        Creates a network for IR prediction with uniform sampling of context
        :param cfg: config
        """
        uniform_context_sampler_cfg = cfg.UniformContextSampler
        pose_enc_cfg = uniform_context_sampler_cfg.PositionalEnc
        memory_net_cfg = uniform_context_sampler_cfg.MemoryNet

        task_cfg = cfg.TASK_CONFIG
        env_cfg = task_cfg.ENVIRONMENT
        sim_cfg = task_cfg.SIMULATOR

        audio_cfg = sim_cfg.AUDIO

        # --------------------------------------------- context encoders ----------------------------------------------------
        assert len(cfg.SENSORS) != 0
        visual_context_enc = VisualEnc(
        )

        if uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry:
            n_input_feats_fusion_context_enc = [0, 0]
            n_input_feats_fusion_context_enc[0] += visual_context_enc.n_out_feats
        else:
            raise ValueError

        audio_context_enc = AudioEnc(
            audio_cfg=cfg.TASK_CONFIG.SIMULATOR.AUDIO,
            log_instead_of_log1p_in_logspace=uniform_context_sampler_cfg.predict_in_logspace and\
                                             uniform_context_sampler_cfg.log_instead_of_log1p_in_logspace,
            log_eps=uniform_context_sampler_cfg.log_gt_eps,
        )
        if uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry:
            n_input_feats_fusion_context_enc[-1] += audio_context_enc.n_out_feats
        else:
            raise ValueError

        pose_context_enc = PositionalEnc(
            positional_enc_cfg=pose_enc_cfg,
        )
        if uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry:
            for n_input_feats_fusion_context_enc_idx in range(len(n_input_feats_fusion_context_enc)):
                n_input_feats_fusion_context_enc[n_input_feats_fusion_context_enc_idx] += pose_context_enc.n_out_feats
        else:
            raise ValueError

        # --------------------------------------- lookup dictionary for modality type tag embedding  ----------------------------------------------------
        if uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry and\
                uniform_context_sampler_cfg.append_modality_type_tag_encoding_to_each_modality_encoding:
            modality_tag_type_lookup_dict = nn.Embedding(len(n_input_feats_fusion_context_enc),
                                                         uniform_context_sampler_cfg.modality_type_tag_encoding_size,)
            for n_input_feats_fusion_context_enc_idx in range(len(n_input_feats_fusion_context_enc)):
                n_input_feats_fusion_context_enc[n_input_feats_fusion_context_enc_idx] +=\
                    uniform_context_sampler_cfg.modality_type_tag_encoding_size
        else:
            raise ValueError

        # --------------------------------------- fusion net for context encodings ----------------------------------------------------
        if uniform_context_sampler_cfg.encode_each_modality_as_independent_context_entry:
            fusion_context_enc = []
            for n_input_feats_fusion_context_enc_idx in range(len(n_input_feats_fusion_context_enc)):
                fusion_context_enc.append(
                    FusionNet(
                        trainer_cfg=uniform_context_sampler_cfg,
                        n_input_feats=n_input_feats_fusion_context_enc[n_input_feats_fusion_context_enc_idx],
                    )
                )
            fusion_context_enc = nn.Sequential(*fusion_context_enc)
        else:
            raise ValueError

        # --------------------------------------------- query encoders ----------------------------------------------------
        n_input_feats_fusion_query_enc = 0
        pose_query_enc = None
        if pose_enc_cfg.shared_pose_encoder_for_context_n_query:
            n_input_feats_fusion_query_enc += pose_context_enc.n_out_feats
            pose_query_enc = pose_context_enc
        else:
            raise ValueError

        # --------------------------------------- fusion net for query encodings ----------------------------------------------------
        fusion_query_enc = FusionNet(
            trainer_cfg=uniform_context_sampler_cfg,
            n_input_feats=n_input_feats_fusion_query_enc,
        )

        # --------------------------------------------- memory net -------------------------------------------------------------
        memory_net = None
        if memory_net_cfg.type == "transformer":
            memory_net = TransformerMemory(
                trainer_cfg=uniform_context_sampler_cfg,
                env_cfg=env_cfg,
            )
        else:
            raise ValueError

        # --------------------------------------- audio decoding ----------------------------------------------------
        audio_dec = AudioDec(
            trainer_cfg=uniform_context_sampler_cfg,
            audio_cfg=audio_cfg,
        )
        # --------------------------------------------------------------------------------------------------------------

        super().__init__(
            visual_context_enc,
            audio_context_enc,
            modality_tag_type_lookup_dict,
            pose_context_enc,
            fusion_context_enc,
            memory_net,
            pose_query_enc,
            fusion_query_enc,
            audio_dec,
            cfg,
        )
