import os
import librosa
import librosa.display
import numpy as np
from pyroomacoustics.experimental.rt60 import measure_rt60
from scipy.io import wavfile

import torch
import torch.nn.functional as F


# -------------------------------------------------- rt60 & edt ---------------------------------------------------
def measure_rtX_wrapper(x, fs=44100, decay_db=60):
    """
    get reverberation time for x dB (time for energy decay by x dB)
    :param x: IR
    :param fs: sampling frequency
    :param decay_db: energy decay in dB
    :return: idx in x where the energy decays by decay_db
    """
    if torch.is_tensor(x):
        x = x.numpy()
    rtX = -1
    while rtX == -1:
        try:
            rtX = measure_rt60(x, fs, decay_db)
        except ValueError:
            if decay_db > 10:
                decay_db -= 10
            else:
                rtX = x.shape[0] / fs

    return rtX


def calculate_rtX_diff(gt, est, fs=44100, decay_db=60, compute_relative_diff=False, get_diff_val=False,
                       get_gt_val=False, get_pred_val=False,):
    """
    compute rtX difference, rtX for gt or rtX for estimated IR
    :param gt: gt IR
    :param est: estimated IR
    :param fs: sampling frequency
    :param decay_db: energy decay in dB
    :param compute_relative_diff:
    :param get_diff_val: flag to get difference in rtX between estimated and gt IR
    :param get_gt_val: flag to get rtX in gt IR
    :param get_pred_val: flag to get rtX in estimated IR
    :return: rtX difference, rtX for estimated or rtX for gt IR
    """
    rtX_gt = measure_rtX_wrapper(gt, fs=fs, decay_db=decay_db)
    rtX_est = measure_rtX_wrapper(est, fs=fs, decay_db=decay_db)
    diff = abs(rtX_gt - rtX_est)
    if compute_relative_diff:
        diff = abs(diff / rtX_gt)

    if get_diff_val:
        return diff
    elif get_gt_val:
        return rtX_gt
    elif get_pred_val:
        return rtX_est


def calculate_edt_diff(gt, est, fs=44100, compute_relative_diff=False):
    """
    get difference in energy decay time (EDT): EDT = RT10 * 6; source: TS-RIR -- https://arxiv.org/pdf/2103.16804.pdf
    :param gt: gt IR
    :param est: estimated IR
    :param fs: sampling frequency
    :param compute_relative_diff: flag to compute relative or absolute difference
    :return: absolute or relative difference in EDT
    """
    rtX_gt = 6 * measure_rtX_wrapper(gt, fs=fs, decay_db=10)
    rtX_est = 6 * measure_rtX_wrapper(est, fs=fs, decay_db=10)
    diff = abs(rtX_gt - rtX_est)
    if compute_relative_diff:
        diff = abs(diff / rtX_gt)
    return diff


# -------------------------------------------------- drr & cte ---------------------------------------------------
def normalize(audio, norm='peak'):
    """
    normalize IR
    :param audio: IR
    :param norm: normalization mode
    :return: normalized IR
    """
    if norm == 'peak':
        peak = abs(audio).max()
        if peak != 0:
            return audio / peak
        else:
            return audio
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError


def measure_drr_energy_ratio(y, cutoff_time=0.003, fs=44100):
    """
    get direct to reverberant energy ratio (DRR)
    :param y: IR
    :param cutoff_time: cutoff time to compute DRR
    :param fs: sampling frequency
    :return: DRR
    """
    direct_sound_idx = int(cutoff_time * fs)

    # removing leading silence
    y = normalize(y)
    y = np.trim_zeros(y, trim='fb')

    # everything up to the given idx is summed up and treated as direct sound energy
    y = np.power(y, 2)
    direct = sum(y[:direct_sound_idx + 1])
    reverberant = sum(y[direct_sound_idx + 1:])
    if direct == 0 or reverberant == 0:
        drr = 1
        # print('Direct or reverberant is 0')
    else:
        drr = 10 * np.log10(direct / reverberant)

    return drr


def calculate_drr_diff(gt, est, cutoff_time=0.003, fs=44100, compute_relative_diff=False, get_diff_val=False,
                       get_gt_val=False, get_pred_val=False,):
    """
    get difference in DRR, DRR for gt or DRR for estimated IR
    :param gt: gt IR
    :param est: estimated IR
    :param cutoff_time: cutoff time to compute DRR
    :param fs: sampling frequency
    :param compute_relative_diff: flag to compute relative difference
    :param get_diff_val: flag to get difference in DRR
    :param get_gt_val: flag to get DRR of gt IR
    :param get_pred_val: flag to get DRR of estimated IR
    :return: difference in DRR, DRR of gt IR or DRR of estimated IR
    """
    drr_gt = measure_drr_energy_ratio(gt, cutoff_time=cutoff_time, fs=fs)
    drr_est = measure_drr_energy_ratio(est, cutoff_time=cutoff_time, fs=fs)
    diff = abs(drr_gt - drr_est)
    if compute_relative_diff:
        diff = abs(diff / drr_gt)

    if get_diff_val:
        return diff
    elif get_gt_val:
        return drr_gt
    elif get_pred_val:
        return drr_est


def measure_lrEnergyRatio(y,):
    """
    get energy ratio b/w left and right channels of IR
    :param y: IR
    :return: energy ratio b/w left and right
    """
    # removing leading silence
    y = normalize(y)
    assert (len(y.shape) == 2) and (y.shape[0] == 2)
    y_l = np.trim_zeros(y[0], trim='fb')
    power_l = np.power(y_l, 2)
    energy_l = np.sum(power_l)

    y_r = np.trim_zeros(y[1], trim='fb')
    power_r = np.power(y_r, 2)
    energy_r = np.sum(power_r)

    lrEnergyRatio = 10 * np.log10((energy_l + 1.0e-8) / (energy_r + 1.0e-8))

    return lrEnergyRatio


def calculate_lrEnergyRatio_diff(gt, est, compute_relative_diff=False, get_diff_val=False, get_gt_val=False,
                                 get_pred_val=False,):
    """
    get difference in lrEnergyRatio, lrEnergyRatio for gt IR or lrEnergyRatio for estimated IR
    :param gt: gt IR
    :param est: estimated IR
    :param compute_relative_diff: flag to compute relative in place of absolute difference
    :param get_diff_val: flag to get lrEnergyRatio difference
    :param get_gt_val: flag to get lrEnergyRatio for gt IR
    :param get_pred_val: flag to get lrEnergyRatio for estimated IR
    :return: lrEnergyRatio difference, lrEnergyRatio for gt IR or lrEnergyRatio for estimated IR
    """
    lrEnergyRatio_gt = measure_lrEnergyRatio(gt)
    lrEnergyRatio_est = measure_lrEnergyRatio(est)
    diff = abs(lrEnergyRatio_gt - lrEnergyRatio_est)
    if compute_relative_diff:
        diff = abs(diff / lrEnergyRatio_gt)

    if get_diff_val:
        return diff
    elif get_gt_val:
        return lrEnergyRatio_gt
    elif get_pred_val:
        return lrEnergyRatio_est


def calculate_cte_diff(gt, est, fs=44100, compute_relative_diff=False,):
    """
    calculate relative difference in CTE, where CTE is ratio of the total sound energy received in the first
    50 ms to the energy received during the rest of the period; source: TS-RIR --  https://arxiv.org/pdf/2103.16804.pdf
                                                                        IR-GAN -- https://arxiv.org/pdf/2010.13219.pdf
    :param gt: gt IR
    :param est: estimated IR
    :param fs: sampling frequency
    :param compute_relative_diff: flag to compute relative in place of absolute difference
    :return: difference in CTE, CTE for gt IR or CTE for estimated IR
    """
    return calculate_drr_diff(gt, est, cutoff_time=0.05, fs=fs, compute_relative_diff=compute_relative_diff,)


def istft(mag_l, phase_l, mag_r=None, phase_r=None, hop_length=172, reconstructed_signal_length=44100):
    """
    computes inverse STFT of a monaural or a binaural spectrogram
    :param mag_l: magnitude of left binaural channel or single mono channel
    :param phase_l: phase of left binaural channel or single mono channel
    :param mag_r: magnitude of right binaural channel
    :param phase_r: phase of right binaural channel
    :param hop_length: hop length for computing ISTFT
    :param reconstructed_signal_length: length of reconstructed IR
    :return: reconstructed IR
    """
    spec_l_complex = mag_l * np.exp(1j * phase_l)

    if len(spec_l_complex.shape) == 3:
        signal_l = []
        for spec_l_complex_entry in spec_l_complex:
            signal_l.append(
                librosa.istft(spec_l_complex_entry, hop_length=hop_length, length=reconstructed_signal_length)
            )
        spec_l_signal = np.stack(signal_l, axis=0)
    elif len(spec_l_complex.shape) == 2:
        spec_l_signal = librosa.istft(spec_l_complex, hop_length=hop_length, length=reconstructed_signal_length)
    else:
        raise ValueError

    signal = [spec_l_signal]

    if mag_r is not None:
        assert phase_r is not None
        spec_r_complex = mag_r * np.exp(1j * phase_r)

        if len(spec_r_complex.shape) == 3:
            signal_r = []
            for spec_r_complex_entry in spec_r_complex:
                signal_r.append(
                    librosa.istft(spec_r_complex_entry, hop_length=hop_length, length=reconstructed_signal_length)
                )
            spec_r_signal = np.stack(signal_r, axis=0)
        elif len(spec_r_complex.shape) == 2:
            spec_r_signal = librosa.istft(spec_r_complex, hop_length=hop_length, length=reconstructed_signal_length)
        else:
            raise ValueError

        signal.append(spec_r_signal)

    # num_channels x B x T -> B x num_channels x T
    return np.transpose(np.stack(signal, axis=0), (1, 0, 2))


def stft_l1_distance(gt_spect_mag=None, gt_spect_phase=None, pred_spect_mag=None, pred_spect_phase=None, mask=None,
                     logspace=False, mag_only=True, eval_mode=False,):
    """
    compute L1 distance between estimated and gt spectrograms
    :param gt_spect_mag: gt spectrogram magnitude
    :param gt_spect_phase: gt spectrogram phase
    :param pred_spect_mag: estimated spectrogram magnitude
    :param pred_spect_phase: estimated spectrogram phase
    :param mask: mask to mark valid entries in batch
    :param logspace: flag to compute loss in log-space
    :param mag_only: flag to compute loss for spectrogram magnitude only
    :param eval_mode: flag to not compute mean along batch dimension
    :return: STFT L1 distance
    """
    if mag_only:
        if logspace:
            gt_spect_mag = torch.log1p(gt_spect_mag)
            pred_spect_mag = torch.log1p(pred_spect_mag)
    else:
        assert not logspace
        assert gt_spect_phase is not None
        assert pred_spect_phase is not None

        gt_spect_real = gt_spect_mag * torch.cos(gt_spect_phase.to(gt_spect_mag.device))
        gt_spect_img = gt_spect_mag * torch.sin(gt_spect_phase.to(gt_spect_mag.device))
        gt_spect_mag = torch.cat([gt_spect_real, gt_spect_img], dim=-1)

        pred_spect_real = pred_spect_mag * torch.cos(pred_spect_phase.to(pred_spect_mag.device))
        pred_spect_img = pred_spect_mag * torch.sin(pred_spect_phase.to(pred_spect_mag.device))
        pred_spect_mag = torch.cat([pred_spect_real, pred_spect_img], dim=-1)

    if mask is not None:
        assert mask.size()[:1] == gt_spect_mag.size()[:1] == pred_spect_mag.size()[:1]
        # pred_spect, gt_spect: B x H x W x C; mask: B
        gt_spect_mag = gt_spect_mag * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pred_spect_mag = pred_spect_mag * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    if mask is None:
        if eval_mode:
            ls_dist = torch.mean(F.l1_loss(pred_spect_mag, gt_spect_mag, reduction='none').view(pred_spect_mag.size(0), -1), dim=-1)
        else:
            ls_dist = F.l1_loss(pred_spect_mag, gt_spect_mag)
    else:
        if eval_mode:
            ls_dist = torch.mean(F.l1_loss(pred_spect_mag, gt_spect_mag, reduction='none').view(pred_spect_mag.size(0), -1), dim=-1)
        else:
            # not counting the contribution from masked out locations in the batch
            ls_dist = torch.sum(torch.abs(pred_spect_mag - gt_spect_mag)) / (torch.sum(mask) * np.prod(list(pred_spect_mag.size())[1:]))

    if eval_mode:
        assert len(ls_dist.size()) == 1
        return ls_dist.tolist()
    else:
        return ls_dist


def gl(spect_mag, hop_length=172, n_fft=511, win_length=0, fs=44100,):
    """
    estimate phase of spectrogram from magnitude using the Griffin-Lim algorithm
    :param spect_mag: spectrogram magnitude
    :param hop_length: hop length for computing spect.
    :param n_fft: number of fft levels for computing spect.
    :param win_length: length of temporal window for computing spect.
    :param fs: sampling frequency
    :return: estimated phase
    """
    # spect_mag: B x H x W x C
    if torch.is_tensor(spect_mag):
        spect_mag = spect_mag.cpu().numpy()

    spect_mag = np.transpose(spect_mag, (0, 3, 1, 2))
    B = spect_mag.shape[0]
    C = spect_mag.shape[1]
    H = spect_mag.shape[2]
    W = spect_mag.shape[3]
    spect_mag = np.reshape(spect_mag, (B * C, H, W))

    recon_imp = librosa.griffinlim(spect_mag,
                                   hop_length=hop_length,
                                   win_length=win_length,
                                   length=fs,)

    fft_windows_recon_imp = librosa.stft(np.asfortranarray(recon_imp),
                                         hop_length=hop_length,
                                         n_fft=n_fft,
                                         win_length=win_length if (win_length != 0) else None,)
    magnitude_recon_imp, phase_recon_imp = librosa.magphase(fft_windows_recon_imp)

    phase_recon_imp = np.angle(phase_recon_imp)
    phase_recon_imp = np.reshape(phase_recon_imp, (B, C, H, W))
    phase_recon_imp = np.transpose(phase_recon_imp, (0, 2, 3, 1))
    return torch.from_numpy(phase_recon_imp)


def compute_spect_metrics(metric_types=[],
                          gt_spect_mag=None,
                          gt_spect_phase=None,
                          pred_spect_mag=None,
                          pred_spect_phase=None,
                          mask=None,
                          eval_mode=False,
                          fs=44100,
                          hop_length=172,
                          n_fft=511,
                          win_length=0,
                          dump_audio_waveforms=False,
                          audio_waveforms_dump_dir=None,
                          start_datapoint_idx_for_batch=0,
                          eval_scenes_this_batch=None,
                          eval_srAzs_this_batch=None,
                          use_gl=False,
                          use_gl_for_gt=False,
                          use_rand_phase=False,
                          use_rand_phase_for_gt=False,
                          ):
    """

    :param metric_types: list of different metrics to be computed
    :param gt_spect_mag: gt spectrogram magnitude
    :param gt_spect_phase: gt spectrogram phase
    :param pred_spect_mag: estimated spectrogram magnitude
    :param pred_spect_phase: estimated spectrogram phase
    :param mask: mask to mark valid idxs in batch
    :param eval_mode: flag that tells if in eval mode or not
    :param fs: sampling frequency
    :param hop_length: hop length for computing spectrograms
    :param n_fft: number of FFT levels for computing spectrograms
    :param win_length: length of temporal window for computing spectrograms
    :param dump_audio_waveforms: flag to dump IR waveforms
    :param audio_waveforms_dump_dir: directory to dump IR waveforms
    :param start_datapoint_idx_for_batch: global index for first entry in batch
    :param eval_scenes_this_batch: environment name for entries in batch
    :param eval_srAzs_this_batch: source, receiver and azimuth for entries in batch
    :param use_gl: use Griffin-Lim for phase estimation
    :param use_gl_for_gt: use Griffin-Lim for phase estimation in gt
    :param use_rand_phase: use random phase
    :param use_rand_phase_for_gt: use random phase in gt
    :return: dict of metric name to values
    """
    metric_name2vals = {}
    gt_signal = None
    pred_signal = None

    if use_gl:
        assert not use_rand_phase

        # pred_spect_mag: B x H x W x C,
        pred_spect_phase = gl(pred_spect_mag, hop_length=hop_length, n_fft=n_fft, win_length=win_length, fs=fs,)

        if use_gl_for_gt:
            # gt_spect_mag: B x H x W x C,
            gt_spect_phase = gl(gt_spect_mag, hop_length=hop_length, n_fft=n_fft, win_length=win_length, fs=fs,)

    if use_rand_phase:
        assert not use_gl

        np.random.seed(42)
        rand_phase = np.random.uniform(-np.pi,
                                       np.pi,
                                       (pred_spect_mag.shape[1],
                                        pred_spect_mag.shape[2]))

        if torch.is_tensor(pred_spect_mag):
            pred_spect_mag_tmp = pred_spect_mag.cpu().numpy()
        else:
            pred_spect_mag_tmp = pred_spect_mag
        pred_spect_phase = np.zeros((pred_spect_mag_tmp.shape[0],
                                     pred_spect_mag_tmp.shape[1],
                                     pred_spect_mag_tmp.shape[2],
                                     pred_spect_mag_tmp.shape[3]))
        pred_spect_phase = np.transpose(pred_spect_phase, (0, 3, 1, 2))
        pred_spect_phase[...] = rand_phase
        pred_spect_phase = np.transpose(pred_spect_phase, (0, 2, 3, 1))
        pred_spect_phase = torch.from_numpy(pred_spect_phase)

        if use_rand_phase_for_gt:
            gt_spect_phase = pred_spect_phase

    if dump_audio_waveforms or ("rel_diff_rt_startFrom60dB" in metric_types) or ("rel_diff_drr_3ms" in metric_types) or\
            ("rel_diff_edt" in metric_types) or ("rel_diff_cte" in metric_types) or\
            ("diff_rt_startFrom60dB" in metric_types) or ("diff_drr_3ms" in metric_types) or\
            ("rel_diff_ratio_lrEnergy" in metric_types) or ("diff_ratio_lrEnergy" in metric_types):
        gt_spect_mag_np = gt_spect_mag
        if torch.is_tensor(gt_spect_mag_np):
            gt_spect_mag_np = gt_spect_mag_np.cpu().numpy()

        assert gt_spect_phase is not None
        gt_spect_phase_np = gt_spect_phase
        if torch.is_tensor(gt_spect_phase_np):
            gt_spect_phase_np = gt_spect_phase_np.cpu().numpy()

        pred_spect_mag_np = pred_spect_mag
        if torch.is_tensor(pred_spect_mag_np):
            pred_spect_mag_np = pred_spect_mag_np.cpu().numpy()

        if pred_spect_phase is not None:
            pred_spect_phase_np = pred_spect_phase
            if torch.is_tensor(pred_spect_phase_np):
                pred_spect_phase_np = pred_spect_phase_np.cpu().numpy()
        else:
            pred_spect_phase_np = gt_spect_phase_np

        # B x N_FFT x N_WINDOWS x NUM_CHANNELS
        assert len(gt_spect_mag_np.shape) == len(gt_spect_phase_np.shape) == len(pred_spect_mag_np.shape) ==\
               len(pred_spect_phase_np.shape) == 4

        if gt_spect_mag.shape[3] == 1:
            assert gt_spect_phase_np.shape[3] == pred_spect_mag_np.shape[3] == pred_spect_phase_np.shape[3] == 1
            gt_signal = istft(mag_l=gt_spect_mag_np[..., 0],
                              phase_l=gt_spect_phase_np[..., 0],
                              hop_length=hop_length,
                              reconstructed_signal_length=fs,)

            pred_signal = istft(mag_l=pred_spect_mag_np[..., 0],
                                phase_l=pred_spect_phase_np[..., 0],
                                hop_length=hop_length,
                                reconstructed_signal_length=fs,)

            assert gt_signal.shape[1] == pred_signal.shape[1] == 1
        else:
            assert gt_spect_phase_np.shape[3] == pred_spect_mag_np.shape[3] == pred_spect_phase_np.shape[3] == 2

            gt_signal = istft(mag_l=gt_spect_mag_np[..., 0],
                              phase_l=gt_spect_phase_np[..., 0],
                              mag_r=gt_spect_mag_np[..., 1],
                              phase_r=gt_spect_phase_np[..., 1],
                              hop_length=hop_length,
                              reconstructed_signal_length=fs,)

            pred_signal = istft(mag_l=pred_spect_mag_np[..., 0],
                                phase_l=pred_spect_phase_np[..., 0],
                                mag_r=pred_spect_mag_np[..., 1],
                                phase_r=pred_spect_phase_np[..., 1],
                                hop_length=hop_length,
                                reconstructed_signal_length=fs,)

            assert gt_signal.shape[1] == pred_signal.shape[1] == 2

        if dump_audio_waveforms:
            assert audio_waveforms_dump_dir is not None
            assert os.path.isdir(audio_waveforms_dump_dir)

            assert eval_scenes_this_batch is not None
            assert eval_srAzs_this_batch is not None

            assert len(eval_scenes_this_batch) == len(eval_srAzs_this_batch)

            for entry_idx, (gt_signal_entry, pred_signal_entry) in enumerate(zip(gt_signal, pred_signal)):
                dump_idx = start_datapoint_idx_for_batch + entry_idx

                assert entry_idx < len(eval_scenes_this_batch)
                scene_this_entry = eval_scenes_this_batch[entry_idx]
                audio_waveforms_dump_dir_this_scene = os.path.join(audio_waveforms_dump_dir, scene_this_entry)
                if not os.path.isdir(audio_waveforms_dump_dir_this_scene):
                    os.makedirs(audio_waveforms_dump_dir_this_scene)

                s_str = f"s{eval_srAzs_this_batch[entry_idx][0]}"
                r_str = f"r{eval_srAzs_this_batch[entry_idx][1]}"
                az_str = f"az{eval_srAzs_this_batch[entry_idx][2]}"

                assert len(gt_signal_entry.shape) == len(pred_signal_entry.shape) == 2

                if gt_signal_entry.shape[0] == pred_signal_entry.shape[0] == 1:
                    gt_signal_entry = np.squeeze(gt_signal_entry, axis=0)
                    pred_signal_entry = np.squeeze(pred_signal_entry, axis=0)

                # NUM_CHANNELS x T -> T x NUM_CHANNELS
                wavfile.write(os.path.join(audio_waveforms_dump_dir_this_scene, f"gt_{dump_idx+1}_{s_str}_{r_str}_{az_str}.wav"),
                              fs, gt_signal_entry.T)
                wavfile.write(os.path.join(audio_waveforms_dump_dir_this_scene, f"pred_{dump_idx+1}_{s_str}_{r_str}_{az_str}.wav"),
                              fs, pred_signal_entry.T)

    for metric_type in metric_types:
        if metric_type == "stft_l1_distance":
            metric_val = stft_l1_distance(
                gt_spect_mag=gt_spect_mag,
                gt_spect_phase=gt_spect_phase,
                pred_spect_mag=pred_spect_mag,
                pred_spect_phase=pred_spect_phase,
                mask=mask,
                logspace=False,
                eval_mode=eval_mode,
                mag_only=((not use_gl) and (not use_rand_phase)),
            )
        else:
            metric_val = []
            for entry_idx, gt_signal_entry in enumerate(gt_signal):
                pred_signal_entry = pred_signal[entry_idx]

                if metric_type == "gt_rt_startFrom60dB":
                    metric_val_entry =\
                        np.mean([calculate_rtX_diff(gt=gt_signal_channel,
                                                    est=pred_signal_channel,
                                                    fs=fs,
                                                    decay_db=60,
                                                    get_gt_val=True,)\
                                 for (gt_signal_channel, pred_signal_channel) in zip(gt_signal_entry, pred_signal_entry)])
                elif metric_type == "pred_rt_startFrom60dB":
                    metric_val_entry =\
                        np.mean([calculate_rtX_diff(gt=gt_signal_channel,
                                                    est=pred_signal_channel,
                                                    fs=fs,
                                                    decay_db=60,
                                                    get_pred_val=True,)\
                                 for (gt_signal_channel, pred_signal_channel) in zip(gt_signal_entry, pred_signal_entry)])
                elif metric_type == "diff_rt_startFrom60dB":
                    metric_val_entry =\
                        np.mean([calculate_rtX_diff(gt=gt_signal_channel,
                                                    est=pred_signal_channel,
                                                    fs=fs,
                                                    decay_db=60,
                                                    get_diff_val=True,)\
                                 for (gt_signal_channel, pred_signal_channel) in zip(gt_signal_entry, pred_signal_entry)])
                elif metric_type == "rel_diff_rt_startFrom60dB":
                    metric_val_entry =\
                        np.mean([calculate_rtX_diff(gt=gt_signal_channel,
                                                    est=pred_signal_channel,
                                                    fs=fs,
                                                    decay_db=60,
                                                    compute_relative_diff=True,
                                                    get_diff_val=True,)\
                                 for (gt_signal_channel, pred_signal_channel) in zip(gt_signal_entry, pred_signal_entry)])
                elif metric_type == "diff_drr_3ms":
                    metric_val_entry =\
                        np.mean([calculate_drr_diff(gt=gt_signal_channel,
                                                    est=pred_signal_channel,
                                                    cutoff_time=3e-3,
                                                    fs=fs,
                                                    get_diff_val=True,)\
                                 for (gt_signal_channel, pred_signal_channel) in zip(gt_signal_entry, pred_signal_entry)])
                elif metric_type == "rel_diff_drr_3ms":
                    metric_val_entry =\
                        np.mean([calculate_drr_diff(gt=gt_signal_channel,
                                                    est=pred_signal_channel,
                                                    cutoff_time=3e-3,
                                                    fs=fs,
                                                    compute_relative_diff=True,
                                                    get_diff_val=True,)\
                                 for (gt_signal_channel, pred_signal_channel) in zip(gt_signal_entry, pred_signal_entry)])
                elif metric_type == "diff_ratio_lrEnergy":
                    metric_val_entry =\
                        calculate_lrEnergyRatio_diff(gt=gt_signal_entry,
                                                     est=pred_signal_entry,
                                                     get_diff_val=True,)
                elif metric_type == "rel_diff_edt":
                    metric_val_entry =\
                        np.mean([calculate_edt_diff(gt=gt_signal_channel,
                                                    est=pred_signal_channel,
                                                    fs=fs,
                                                    compute_relative_diff=True,)\
                                 for (gt_signal_channel, pred_signal_channel) in zip(gt_signal_entry, pred_signal_entry)])
                elif metric_type == "rel_diff_cte":
                    metric_val_entry =\
                        np.mean([calculate_cte_diff(gt=gt_signal_channel,
                                                    est=pred_signal_channel,
                                                    fs=fs,
                                                    compute_relative_diff=True,)\
                                 for (gt_signal_channel, pred_signal_channel) in zip(gt_signal_entry, pred_signal_entry)])

                else:
                    raise ValueError

                metric_val.append(metric_val_entry)

            if not eval_mode:
                metric_val = np.mean(metric_val)

        metric_name2vals[metric_type] = metric_val

    return metric_name2vals
