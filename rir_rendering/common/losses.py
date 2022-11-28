import numpy as np
from pyroomacoustics.experimental.rt60 import measure_rt60

import torch
import torch.nn as nn
import torch.nn.functional as F


def stft_l1_loss(gt_spect=None, pred_spect=None, mask=None, logspace=True, log1p_gt=False,
                 log_instead_of_log1p_in_logspace=False,  log_gt=False, log_gt_eps=1.0e-8, mag_only=True,):
    """
    compute L1 loss between gt and estimated spectrograms (spect.)
    :param gt_spect: gt spect.
    :param pred_spect: estimated spect.
    :param mask: mask to mark valid entries in batch
    :param logspace: flag to tell if spect. estimation in log-space or not
    :param log1p_gt: flag to decide to log(1 + gt_spect) or not
    :param log_instead_of_log1p_in_logspace: flag to decide to log(gt_spect/pred_spect) instead of log(1 + gt_spect/pred_spect)
    :param log_gt: flag to decide to log(gt_spect)
    :param log_gt_eps: eps to be added before computing log for numerical stability
    :param mag_only: flag is set if spect. is magnitude only
    :return: L1 loss between gt and estimated spects.
    """
    if mag_only:
        assert torch.all(gt_spect >= 0.).item(), "mag_only"

        if logspace:
            if log_instead_of_log1p_in_logspace:
                if log_gt:
                    gt_spect = torch.log(gt_spect + log_gt_eps)
                else:
                    pred_spect = torch.exp(pred_spect) - log_gt_eps
            else:
                if log1p_gt:
                    gt_spect = torch.log1p(gt_spect)
                else:
                    pred_spect = torch.exp(pred_spect) - 1

        if mask is not None:
            assert mask.size()[:1] == gt_spect.size()[:1] == pred_spect.size()[:1]
            # pred_spect, gt_spect: B x H x W x C; mask: B
            gt_spect = gt_spect * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            pred_spect = pred_spect * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError

    if mask is None:
        loss = F.l1_loss(pred_spect, gt_spect)
    else:
        # not counting the contribution from masked out locations in the batch
        loss = torch.sum(torch.abs(pred_spect - gt_spect)) / (torch.sum(mask) * np.prod(list(pred_spect.size())[1:]))

    return loss


def compute_spect_losses(loss_types=[],
                         loss_weights=[],
                         gt_spect=None,
                         pred_spect=None,
                         mask=None,
                         logspace=True,
                         log1p_gt=False,
                         log_instead_of_log1p_in_logspace=False,
                         log_gt=False,
                         log_gt_eps=1.0e-8,):
    """
    get spectrogram (spect.) loss (error in spect. estimation)
    :param loss_types: loss type
    :param loss_weights: loss weight
    :param gt_spect: gt IR spect.
    :param pred_spect: estimated IR spec.
    :param mask: mask to mark valid entries in batch
    :param logspace: flag to tell if spect. estimation in log-space or not
    :param log1p_gt: flag to decide to log(1 + gt_spect) or not
    :param log_instead_of_log1p_in_logspace: flag to decide to log(gt_spect/pred_spect) instead of log(1 + gt_spect/pred_spect)
    :param log_gt: flag to decide to log(gt_spect)
    :param log_gt_eps: eps to be added before computing log for numerical stability
    :return: spect. loss
    """
    loss = 0.
    for loss_idx, loss_type in enumerate(loss_types):
        if loss_type == "stft_l1_loss":
            loss += (stft_l1_loss(
                gt_spect=gt_spect,
                pred_spect=pred_spect,
                mask=mask,
                logspace=logspace,
                log1p_gt=log1p_gt,
                log_instead_of_log1p_in_logspace=log_instead_of_log1p_in_logspace,
                log_gt=log_gt,
                log_gt_eps=log_gt_eps,
            ) * loss_weights[loss_idx])
        else:
            raise ValueError

    return loss


def compute_spect_energy_decay_losses(loss_type="l1_loss",
                                      loss_weight=1.0,
                                      gts=None,
                                      preds=None,
                                      mask=None,
                                      slice_till_direct_signal=False,
                                      direct_signal_len_in_ms=50,
                                      dont_collapse_across_freq_dim=False,
                                      sr=16000,
                                      hop_length=62,
                                      win_length=248,
                                      ):
    """
    compute energy decay loss
    :param loss_type: loss type
    :param loss_weight: loss weight
    :param gts: gt IRs
    :param preds: estimated IRs
    :param mask: mask to mark valid entries in batch
    :param slice_till_direct_signal: remove direct signal part of IR
    :param direct_signal_len_in_ms: direct signal length in milliseconds
    :param dont_collapse_across_freq_dim: collapse along frequency dimension of spectrogram (spect.)
    :param sr: sampling rate
    :param hop_length: hop length to compute spect.
    :param win_length: length of temporal window to compute spect.
    :return: energy decay loss
    """
    assert len(gts.size()) == len(preds.size()) == 4
    assert gts.size(-1) in [1, 2]
    assert preds.size(-1) in [1, 2]

    slice_idx = None
    if slice_till_direct_signal:
        if direct_signal_len_in_ms == 50:
            if (sr == 16000) and (hop_length == 62) and (win_length == 248):
                # (62 * 11 + 248 / 2) / 16000 = 0.050375 (50 ms)
                # so has to use the 12th window (idx = 11).. so slice idx should be 12
                slice_idx = 12
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    if slice_till_direct_signal:
        if dont_collapse_across_freq_dim:
            gts_fullBandAmpEnv = gts[:slice_idx]
        else:
            gts_fullBandAmpEnv = torch.sum(gts[:slice_idx], dim=-3)
    else:
        if dont_collapse_across_freq_dim:
            gts_fullBandAmpEnv = gts
        else:
            gts_fullBandAmpEnv = torch.sum(gts, dim=-3)
    power_gts_fullBandAmpEnv = gts_fullBandAmpEnv ** 2
    energy_gts_fullBandAmpEnv = torch.flip(torch.cumsum(torch.flip(power_gts_fullBandAmpEnv, [-2]), -2), [-2])
    valid_loss_idxs = ((energy_gts_fullBandAmpEnv != 0.).type(energy_gts_fullBandAmpEnv.dtype))[..., 1:, :]

    db_gts_fullBandAmpEnv = 10 * torch.log10(energy_gts_fullBandAmpEnv + 1.0e-13)
    norm_db_gts_fullBandAmpEnv = db_gts_fullBandAmpEnv - db_gts_fullBandAmpEnv[..., :1, :]
    norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv[..., 1:, :]
    if slice_till_direct_signal:
        weighted_norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv
    else:
        weighted_norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv * valid_loss_idxs

    if slice_till_direct_signal:
        if dont_collapse_across_freq_dim:
            preds_fullBandAmpEnv = preds[:slice_idx]
        else:
            preds_fullBandAmpEnv = torch.sum(preds[:slice_idx], dim=-3)
    else:
        if dont_collapse_across_freq_dim:
            preds_fullBandAmpEnv = preds
        else:
            preds_fullBandAmpEnv = torch.sum(preds, dim=-3)
    power_preds_fullBandAmpEnv = preds_fullBandAmpEnv ** 2
    energy_preds_fullBandAmpEnv = torch.flip(torch.cumsum(torch.flip(power_preds_fullBandAmpEnv, [-2]), -2), [-2])
    db_preds_fullBandAmpEnv = 10 * torch.log10(energy_preds_fullBandAmpEnv + 1.0e-13)
    norm_db_preds_fullBandAmpEnv = db_preds_fullBandAmpEnv - db_preds_fullBandAmpEnv[..., :1, :]
    norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv[..., 1:, :]
    if slice_till_direct_signal:
        weighted_norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv
    else:
        weighted_norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv * valid_loss_idxs

    if loss_type == "l1_loss":
        if mask is None:
            loss = F.l1_loss(weighted_norm_db_preds_fullBandAmpEnv, weighted_norm_db_gts_fullBandAmpEnv)
        else:
            # not counting the contribution from masked out locations in the batch
            assert torch.sum(mask) == mask.size(0)
            loss = torch.sum(torch.abs(weighted_norm_db_preds_fullBandAmpEnv - weighted_norm_db_gts_fullBandAmpEnv)) /\
                   (torch.sum(mask) * np.prod(list(weighted_norm_db_preds_fullBandAmpEnv.size())[1:]))
    else:
        raise NotImplementedError

    loss = loss * loss_weight

    return loss

