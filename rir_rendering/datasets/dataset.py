import copy
import os
import pickle
import math
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import librosa
import soundfile as sf
from PIL import Image, ImageEnhance
import cv2
from skimage.measure import block_reduce

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis

from rir_rendering.common.eval_metrics import calculate_drr_diff, calculate_rtX_diff


SCENE_NAME_TO_IDX = {
    "mp3d":
        {'sT4fr6TAbpF': 0, 'E9uDoFAP3SH': 1, 'VzqfbhrpDEA': 2, 'kEZ7cmS4wCh': 3, '29hnd4uzFmX': 4, 'ac26ZMwG7aT': 5,
         's8pcmisQ38h': 6, 'rPc6DW4iMge': 7, 'EDJbREhghzL': 8, 'mJXqzFtmKg4': 9, 'B6ByNegPMKs': 10, 'JeFG25nYj2p': 11,
         '82sE5b5pLXE': 12, 'D7N2EKCX4Sj': 13, '7y3sRwLe3Va': 14, '5LpN3gDmAk7': 15, 'gTV8FGcVJC9': 16, 'ur6pFq6Qu1A': 17,
         'qoiz87JEwZ2': 18, 'PuKPg4mmafe': 19, 'VLzqgDo317F': 20, 'aayBHfsNo7d': 21, 'JmbYfDe2QKZ': 22, 'XcA2TqTSSAj': 23,
         '8WUmhLawc2A': 24, 'sKLMLpTHeUy': 25, 'r47D5H71a5s': 26, 'Uxmj2M2itWa': 27, 'Pm6F8kyY3z2': 28, 'p5wJjkQkbXX': 29,
         '759xd9YjKW5': 30, 'JF19kD82Mey': 31, 'V2XKFyX4ASd': 32, '1LXtFkjw3qL': 33, '17DRP5sb8fy': 34, '5q7pvUzZiYa': 35,
         'VVfe2KiqLaN': 36, 'Vvot9Ly1tCj': 37, 'ULsKaCPVFJR': 38, 'D7G3Y4RVNrH': 39, 'uNb9QFRL6hY': 40, 'ZMojNkEp431': 41,
         '2n8kARJN3HM': 42, 'vyrNrziPKCB': 43, 'e9zR4mvMWw7': 44, 'r1Q1Z4BcV1o': 45, 'PX4nDJXEHrG': 46, 'YmJkqBEsHnH': 47,
         'b8cTxDM8gDG': 48, 'GdvgFV5R1Z5': 49, 'pRbA3pwrgk9': 50, 'jh4fc5c5qoQ': 51, '1pXnuDYAj8r': 52, 'S9hNv5qa7GM': 53,
         'VFuaQ6m2Qom': 54, 'cV4RVeZvu5T': 55, 'SN83YJsR3w2': 56, '2azQ1b91cZZ': 57, '5ZKStnWn8Zo': 58, '8194nk5LbLH': 59,
         'ARNzJeq3xxb': 60, 'EU6Fwq7SyZv': 61, 'QUCTc6BB5sX': 62, 'TbHJrupSAjP': 63, 'UwV83HsGsw3': 64, 'Vt2qJdWjCF2': 65,
         'WYY7iVyf5p8': 66, 'X7HyMhZNoso': 67, 'YFuZgdQ5vWj': 68, 'Z6MFQCViBuw': 69, 'fzynW3qQPVF': 70, 'gYvKGZ5eRqb': 71,
         'gxdoqLR6rwA': 72, 'jtcxE69GiFV': 73, 'oLBMNvg9in8': 74, 'pLe4wQe7qrG': 75, 'pa4otMbVnkk': 76, 'q9vSo1VnCiC': 77,
         'rqfALeAoiTq': 78, 'wc2JMjhGNzB': 79, 'x8F5xyUWy9e': 80, 'yqstnuAEVhm': 81, 'zsNo4HB9uLZ': 82},
}

SCENE_SPLITS = {
    "mp3d":
        {
            'train':
                ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX', 'ac26ZMwG7aT',
                 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4', 'B6ByNegPMKs', 'JeFG25nYj2p',
                 '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va', '5LpN3gDmAk7', 'gTV8FGcVJC9', 'ur6pFq6Qu1A',
                 'qoiz87JEwZ2', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d', 'JmbYfDe2QKZ', 'XcA2TqTSSAj',
                 '8WUmhLawc2A', 'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa', 'Pm6F8kyY3z2', 'p5wJjkQkbXX',
                 '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL', '17DRP5sb8fy', '5q7pvUzZiYa',
                 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH', 'uNb9QFRL6hY', 'ZMojNkEp431',
                 '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7', 'r1Q1Z4BcV1o', 'PX4nDJXEHrG', 'YmJkqBEsHnH',
                 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9', 'jh4fc5c5qoQ', '1pXnuDYAj8r', 'S9hNv5qa7GM',
                 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'SN83YJsR3w2',],
            'unseen_eval':
                ['2azQ1b91cZZ', '5ZKStnWn8Zo', '8194nk5LbLH', 'ARNzJeq3xxb', 'EU6Fwq7SyZv', 'QUCTc6BB5sX',
                 'TbHJrupSAjP', 'UwV83HsGsw3', 'Vt2qJdWjCF2', 'WYY7iVyf5p8', 'X7HyMhZNoso', 'YFuZgdQ5vWj',
                 'Z6MFQCViBuw', 'fzynW3qQPVF', 'gYvKGZ5eRqb', 'gxdoqLR6rwA', 'jtcxE69GiFV', 'oLBMNvg9in8',
                 'pLe4wQe7qrG', 'pa4otMbVnkk', 'q9vSo1VnCiC', 'rqfALeAoiTq', 'wc2JMjhGNzB', 'x8F5xyUWy9e',
                 'yqstnuAEVhm', 'zsNo4HB9uLZ'],
        },
}


class UniformContextSamplerDataset(Dataset):
    def __init__(self, split="train", all_scenes_graphs_this_split=None, cfg=None, all_scenes_observations=None,
                 eval_mode=False, ckpt_rootdir_path=None,):
        """
        Dataset class
        :param split: train / eval / unseen_eval split
        :param all_scenes_graphs_this_split: all scene graphs in this split
        :param cfg: config
        :param all_scenes_observations: cached observations for all scenes
        :param eval_mode: flag to decide if currently in eval_mode
        :param ckpt_rootdir_path: current test directory path for dumping pickle files with eval scene names, input IR
                                pose indexes and query IR pose indexes
        """

        self.split = split
        self.config = cfg
        task_cfg = cfg.TASK_CONFIG
        self.sim_cfg = task_cfg.SIMULATOR
        self.env_cfg = task_cfg.ENVIRONMENT
        self.task_cfg = task_cfg.TASK
        self.audio_cfg = self.sim_cfg.AUDIO
        self.uniform_context_sampler_cfg = cfg.UniformContextSampler

        self._eval_mode = eval_mode
        if eval_mode:
            self._all_eval_datapoints_sampledEchoPoseIdxs = []
            self._all_eval_datapoints_sceneNames = []
            self._all_eval_datapoints_sampledArbitraryPoseIdxs = []

        self.scene_dataset = self.sim_cfg.SCENE_DATASET

        self.rir_sampling_rate = self.audio_cfg.RIR_SAMPLING_RATE

        self.hop_length = self.audio_cfg.HOP_LENGTH
        self.n_fft = self.audio_cfg.N_FFT
        self.win_length = self.audio_cfg.WIN_LENGTH

        assert "BIN_SPECT_MAG_SENSOR" in self.task_cfg.SENSORS
        self._echo_feat_shape = self.task_cfg.BIN_SPECT_MAG_SENSOR.FEATURE_SHAPE

        assert "POSE_SENSOR" in self.task_cfg.SENSORS
        self._pose_feat_shape = self.task_cfg.POSE_SENSOR.FEATURE_SHAPE

        assert os.path.isfile(self.audio_cfg.VALID_ECHO_POSES_PATH)
        with open(self.audio_cfg.VALID_ECHO_POSES_PATH, "rb") as fi:
            self._arr_echo_poses_per_scene = pickle.load(fi)

        assert split in ["train", "seen_eval", "unseen_eval"]

        self.max_context_length = self.env_cfg.MAX_CONTEXT_LENGTH
        self._is_max_context_length_zero = False
        if self.max_context_length == 0:
            self.max_context_length = 1
            self._is_max_context_length_zero = True
        self.max_query_length = self.env_cfg.MAX_QUERY_LENGTH

        self.load_query_for_arbitrary_rirs_from_disk = self.env_cfg.LOAD_QUERY_FOR_ARBITRARY_RIRS_FROM_DISK and \
                                                       (split in ["seen_eval", "unseen_eval"])
        self.load_context_from_disk = self.env_cfg.LOAD_CONTEXT_FROM_DISK and self.load_query_for_arbitrary_rirs_from_disk
        self.context_pose_idxs_from_disk = None
        self.arbitrary_rir_query_pose_idxs_from_disk = None
        self.arbitrary_rir_query_pose_subgraph_idxs_from_disk = None
        self.arbitrary_rir_scene_names_from_disk = None
        self.eval_datapoint_count = 0
        if self.load_query_for_arbitrary_rirs_from_disk:
            if split == "seen_eval":
                arbitrary_rir_seen_env_eval_query_pose_idxs_path = self.env_cfg.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH
                assert arbitrary_rir_seen_env_eval_query_pose_idxs_path is not None
                assert os.path.isfile(arbitrary_rir_seen_env_eval_query_pose_idxs_path)
                with open(arbitrary_rir_seen_env_eval_query_pose_idxs_path, "rb") as fi:
                    self.arbitrary_rir_query_pose_idxs_from_disk = pickle.load(fi)

                arbitrary_rir_seen_env_eval_scene_names_path = self.env_cfg.ARBITRARY_RIR_SEEN_ENV_EVAL_SCENE_NAMES_PATH
                assert arbitrary_rir_seen_env_eval_scene_names_path is not None
                assert os.path.isfile(arbitrary_rir_seen_env_eval_scene_names_path)
                with open(arbitrary_rir_seen_env_eval_scene_names_path, "rb") as fi:
                    self.arbitrary_rir_scene_names_from_disk = pickle.load(fi)

                arbitrary_rir_seen_env_eval_query_pose_subgraph_idxs_path = self.env_cfg.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH
                assert arbitrary_rir_seen_env_eval_query_pose_subgraph_idxs_path is not None
                assert os.path.isfile(arbitrary_rir_seen_env_eval_query_pose_subgraph_idxs_path)
                with open(arbitrary_rir_seen_env_eval_query_pose_subgraph_idxs_path, "rb") as fi:
                    self.arbitrary_rir_query_pose_subgraph_idxs_from_disk = pickle.load(fi)
            elif split == "unseen_eval":
                arbitrary_rir_unseen_env_eval_query_pose_idxs_path = self.env_cfg.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH
                assert arbitrary_rir_unseen_env_eval_query_pose_idxs_path is not None
                assert os.path.isfile(arbitrary_rir_unseen_env_eval_query_pose_idxs_path)
                with open(arbitrary_rir_unseen_env_eval_query_pose_idxs_path, "rb") as fi:
                    self.arbitrary_rir_query_pose_idxs_from_disk = pickle.load(fi)

                arbitrary_rir_unseen_env_val_scene_names_path = self.env_cfg.ARBITRARY_RIR_UNSEEN_ENV_EVAL_SCENE_NAMES_PATH
                assert arbitrary_rir_unseen_env_val_scene_names_path is not None
                assert os.path.isfile(arbitrary_rir_unseen_env_val_scene_names_path)
                with open(arbitrary_rir_unseen_env_val_scene_names_path, "rb") as fi:
                    self.arbitrary_rir_scene_names_from_disk = pickle.load(fi)

                arbitrary_rir_unseen_env_val_query_pose_subgraph_idxs_path = self.env_cfg.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH
                assert arbitrary_rir_unseen_env_val_query_pose_subgraph_idxs_path is not None
                assert os.path.isfile(arbitrary_rir_unseen_env_val_query_pose_subgraph_idxs_path)
                with open(arbitrary_rir_unseen_env_val_query_pose_subgraph_idxs_path, "rb") as fi:
                    self.arbitrary_rir_query_pose_subgraph_idxs_from_disk = pickle.load(fi)

            if self.load_context_from_disk:
                if split == "seen_eval":
                    seen_env_eval_context_pose_idxs_path = self.env_cfg.SEEN_ENV_EVAL_CONTEXT_POSE_IDXS_PATH
                    assert seen_env_eval_context_pose_idxs_path is not None
                    assert os.path.isfile(seen_env_eval_context_pose_idxs_path)
                    with open(seen_env_eval_context_pose_idxs_path, "rb") as fi:
                        self.context_pose_idxs_from_disk = pickle.load(fi)

                elif split == "unseen_eval":
                    unseen_env_eval_context_pose_idxs_path = self.env_cfg.UNSEEN_ENV_EVAL_CONTEXT_POSE_IDXS_PATH
                    assert unseen_env_eval_context_pose_idxs_path is not None
                    assert os.path.isfile(unseen_env_eval_context_pose_idxs_path)
                    with open(unseen_env_eval_context_pose_idxs_path, "rb") as fi:
                        self.context_pose_idxs_from_disk = pickle.load(fi)

        self._arr_query_poses_per_scene = None
        self._array_context_poses_per_scene = None
        if split == "train":
            assert os.path.isfile(self.audio_cfg.VALID_ARBITRARY_RIR_TRAIN_POSES_PATH)
            with open(self.audio_cfg.VALID_ARBITRARY_RIR_TRAIN_POSES_PATH, "rb") as fi:
                self._arr_query_poses_per_scene = pickle.load(fi)
        elif split == "seen_eval":
            assert os.path.isfile(self.audio_cfg.VALID_ARBITRARY_RIR_SEEN_ENV_EVAL_POSES_PATH)
            with open(self.audio_cfg.VALID_ARBITRARY_RIR_SEEN_ENV_EVAL_POSES_PATH, "rb") as fi:
                self._arr_query_poses_per_scene = pickle.load(fi)
        elif split == "unseen_eval":
            assert os.path.isfile(self.audio_cfg.VALID_ARBITRARY_RIR_UNSEEN_ENV_EVAL_POSES_PATH)
            with open(self.audio_cfg.VALID_ARBITRARY_RIR_UNSEEN_ENV_EVAL_POSES_PATH, "rb") as fi:
                self._arr_query_poses_per_scene = pickle.load(fi)

        self._all_scenes_graphs_this_split = all_scenes_graphs_this_split

        assert all_scenes_observations is not None
        self.all_scenes_observations = all_scenes_observations

        self.binaural_rir_dir = self.audio_cfg.RIR_DIR
        assert os.path.isdir(self.binaural_rir_dir)

        self.sweep_audio_file_path = os.path.join(self.audio_cfg.SWEEP_AUDIO_DIR, self.audio_cfg.SWEEP_AUDIO_FILENAME)
        assert os.path.isfile(self.sweep_audio_file_path)
        self.sweep_audio = None
        self.load_sweep_audio()

        self.scenes_in_split = SCENE_SPLITS[self.scene_dataset]["unseen_eval" if (split == "unseen_eval") else "train"]

        self.num_datapoints_per_epoch = (self.uniform_context_sampler_cfg.num_datapoints_per_scene_train * len(self.scenes_in_split))\
            if (split == "train") else (self.uniform_context_sampler_cfg.num_datapoints_per_scene_eval * len(self.scenes_in_split))

        self.ckpt_rootdir_path = ckpt_rootdir_path
        if eval_mode:
            assert self.ckpt_rootdir_path is not None
            assert os.path.isdir(self.ckpt_rootdir_path)

    def dump_eval_pkls(self):
        """
        dump pickle files containing eval scene names, input IR pose indexes and query IR pose indexes
        :return: None
        """
        assert self._eval_mode

        with open(os.path.join(self.ckpt_rootdir_path,
                               f"{self.split}_{self.num_datapoints_per_epoch}datapoints_sceneNames.pkl"), "wb") as fo:
            pickle.dump(self._all_eval_datapoints_sceneNames, fo, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.ckpt_rootdir_path,
                               f"{self.split}_{self.num_datapoints_per_epoch}datapoints_sampledEchoPoseIdxs.pkl"), "wb")\
                as fo:
            pickle.dump(self._all_eval_datapoints_sampledEchoPoseIdxs, fo, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.ckpt_rootdir_path,
                               f"{self.split}_{self.num_datapoints_per_epoch}datapoints_sampledArbitraryPoseIdxs.pkl"),
                  "wb") as fo:
            pickle.dump(self._all_eval_datapoints_sampledArbitraryPoseIdxs, fo, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return self.num_datapoints_per_epoch

    def __getitem__(self, item):
        this_datapoint = self._get_datapoint(item)

        context_views_this_datapoint = torch.from_numpy(this_datapoint["context"]["views"])
        context_echoes_mag_this_datapoint = torch.from_numpy(this_datapoint["context"]["echoes_mag"])
        context_poses_this_datapoint = torch.from_numpy(this_datapoint["context"]["poses"])
        context_mask_this_datapoint = torch.from_numpy(this_datapoint["context"]["mask"])

        gt_queryImpEchoes_mag_this_datapoint = torch.from_numpy(this_datapoint["query"]["gt_impEchoes_mag"])
        if self._eval_mode:
            gt_queryImpEchoes_phase_this_datapoint = torch.from_numpy(this_datapoint["query"]["gt_impEchoes_phase"])
        query_poses_this_datapoint = torch.from_numpy(this_datapoint["query"]["poses"])

        query_mask_this_datapoint = torch.from_numpy(this_datapoint["query"]["mask"])
        query_scene_idxs_this_datapoint = torch.from_numpy(this_datapoint["query"]["scene_idxs"])
        if self._eval_mode:
            query_scene_srAzs_this_datapoint = torch.from_numpy(this_datapoint["query"]["scene_srAzs"])

        if self._eval_mode:
            rtrn_lst = [context_views_this_datapoint, context_echoes_mag_this_datapoint, context_poses_this_datapoint,\
                        context_mask_this_datapoint, gt_queryImpEchoes_mag_this_datapoint, gt_queryImpEchoes_phase_this_datapoint,\
                        query_poses_this_datapoint, query_mask_this_datapoint, query_scene_idxs_this_datapoint, query_scene_srAzs_this_datapoint]
        else:
            rtrn_lst = [context_views_this_datapoint, context_echoes_mag_this_datapoint, context_poses_this_datapoint,\
                        context_mask_this_datapoint, gt_queryImpEchoes_mag_this_datapoint, query_poses_this_datapoint,\
                        query_mask_this_datapoint, query_scene_idxs_this_datapoint]

        return rtrn_lst

    def load_sweep_audio(self):
        """
        load anechoic audio for sinusoidal sweep
        :return: None
        """
        sweep_audio, fs = sf.read(self.sweep_audio_file_path, dtype='int16')
        if fs != self.rir_sampling_rate:
            sweep_audio = librosa.resample(sweep_audio.astype("float32"), fs, self.rir_sampling_rate).astype("int16")

        self.sweep_audio = librosa.util.fix_length(sweep_audio, self.rir_sampling_rate)

    def _get_datapoint(self, item_):
        """
        get datapoint given datapoint index (datapoint matters only for eval)
        :param item_: datapoint index
        :return: datapoint
        """
        if self.load_query_for_arbitrary_rirs_from_disk:
            assert item_ < len(self.arbitrary_rir_scene_names_from_disk)
            datapoint_scene = self.arbitrary_rir_scene_names_from_disk[item_]

            assert item_ < len(self.arbitrary_rir_query_pose_subgraph_idxs_from_disk)
            datapoint_subgraph_idx = int(self.arbitrary_rir_query_pose_subgraph_idxs_from_disk[item_])
        else:
            datapoint_scene_idx = torch.randint(len(self.scenes_in_split), (1,)).item()
            datapoint_scene = self.scenes_in_split[datapoint_scene_idx]

            assert datapoint_scene in self._arr_query_poses_per_scene
            datapoint_subgraph_idx_idx = torch.randint(len(self._arr_query_poses_per_scene[datapoint_scene]), (1,)).item()
            datapoint_subgraph_idx = int(list(self._arr_query_poses_per_scene[datapoint_scene].keys())[datapoint_subgraph_idx_idx])

        if self._eval_mode:
            self._all_eval_datapoints_sceneNames += ([datapoint_scene] * self.max_query_length)

        assert datapoint_scene in self._arr_echo_poses_per_scene
        assert datapoint_subgraph_idx is not None
        assert datapoint_subgraph_idx in self._arr_echo_poses_per_scene[datapoint_scene]
        self._arr_echo_poses_per_scene[datapoint_scene][datapoint_subgraph_idx] =\
            np.array(self._arr_echo_poses_per_scene[datapoint_scene][datapoint_subgraph_idx])
        num_scene_context_poses = len(self._arr_echo_poses_per_scene[datapoint_scene][datapoint_subgraph_idx])

        max_context_length_this_datapoint = self.max_context_length
        num_context_poses = max_context_length_this_datapoint
        if num_context_poses > num_scene_context_poses:
            num_context_poses = num_scene_context_poses

        if self.load_context_from_disk:
            assert item_ < len(self.context_pose_idxs_from_disk)
            assert max_context_length_this_datapoint <= len(self.context_pose_idxs_from_disk[item_]),\
                print(max_context_length_this_datapoint, len(self.context_pose_idxs_from_disk[item_]))
            context_pose_idxs = self.context_pose_idxs_from_disk[item_][:max_context_length_this_datapoint]
            assert isinstance(context_pose_idxs, list)
        else:
            if num_scene_context_poses != num_context_poses:
                context_pose_idxs = torch.multinomial(torch.ones(num_scene_context_poses) / num_scene_context_poses,
                                                      num_context_poses).tolist()
            else:
                context_pose_idxs = torch.multinomial(torch.ones(num_scene_context_poses) / num_scene_context_poses,
                                                      num_scene_context_poses).tolist()
                if max_context_length_this_datapoint - num_scene_context_poses > 0:
                    context_pose_idxs += torch.multinomial(torch.ones(num_scene_context_poses) / num_scene_context_poses,
                                                           max_context_length_this_datapoint - num_scene_context_poses,
                                                           replacement=True).tolist()

        context_poses = self._arr_echo_poses_per_scene[datapoint_scene][datapoint_subgraph_idx][context_pose_idxs, :].tolist()

        assert datapoint_scene in self._arr_query_poses_per_scene
        assert datapoint_subgraph_idx is not None
        assert datapoint_subgraph_idx in self._arr_query_poses_per_scene[datapoint_scene]
        self._arr_query_poses_per_scene[datapoint_scene][datapoint_subgraph_idx] =\
            np.array(self._arr_query_poses_per_scene[datapoint_scene][datapoint_subgraph_idx])

        num_scene_arbitrary_rir_poses = self._arr_query_poses_per_scene[datapoint_scene][datapoint_subgraph_idx].shape[0]

        if self.load_query_for_arbitrary_rirs_from_disk:
            assert item_ < len(self.arbitrary_rir_query_pose_idxs_from_disk)
            query_pose_idxs = self.arbitrary_rir_query_pose_idxs_from_disk[item_][:self.max_query_length]
            assert isinstance(query_pose_idxs, list)
        else:
            if self.max_query_length > num_scene_arbitrary_rir_poses:
                query_pose_idxs = torch.multinomial(torch.ones(num_scene_arbitrary_rir_poses) / num_scene_arbitrary_rir_poses,
                                                    num_scene_arbitrary_rir_poses).tolist()
                query_pose_idxs += torch.multinomial(torch.ones(num_scene_arbitrary_rir_poses) / num_scene_arbitrary_rir_poses,
                                                     self.max_query_length - num_scene_arbitrary_rir_poses,
                                                     replacement=True).tolist()
            else:
                query_pose_idxs = torch.multinomial(torch.ones(num_scene_arbitrary_rir_poses) / num_scene_arbitrary_rir_poses,
                                                    self.max_query_length).tolist()

        query_poses = self._arr_query_poses_per_scene[datapoint_scene][datapoint_subgraph_idx][query_pose_idxs, :].tolist()

        if self._eval_mode:
            self._all_eval_datapoints_sampledEchoPoseIdxs += context_pose_idxs
            self._all_eval_datapoints_sampledArbitraryPoseIdxs += query_pose_idxs

        datapoint = {}

        view_sensor_height = None
        view_sensor_width = None
        view_sensor_nChannels = None
        assert len(self.config.SENSORS) == 2
        if self.config.SENSORS in [["RGB_SENSOR", "DEPTH_SENSOR"], ["DEPTH_SENSOR", "RGB_SENSOR"]]:
            assert self.sim_cfg.RGB_SENSOR.HEIGHT == self.sim_cfg.DEPTH_SENSOR.HEIGHT
            view_sensor_height = self.sim_cfg.RGB_SENSOR.HEIGHT

            assert self.sim_cfg.RGB_SENSOR.WIDTH == self.sim_cfg.DEPTH_SENSOR.WIDTH
            view_sensor_width = self.sim_cfg.RGB_SENSOR.WIDTH

            view_sensor_nChannels = 4
        else:
            raise ValueError

        context_views_this_datapoint = np.zeros((self.max_context_length,
                                                view_sensor_height,
                                                view_sensor_width,
                                                view_sensor_nChannels)).astype("float32")

        assert self._pose_feat_shape[0] == 5
        context_poses_this_datapoint = np.zeros((self.max_context_length, 5)).astype("float32")
        query_poses_this_datapoint = np.zeros((self.max_query_length, 5)).astype("float32")

        context_mask_this_datapoint = np.zeros(self.max_context_length).astype("uint8")
        query_mask_this_datapoint = np.zeros(self.max_query_length).astype("uint8")
        query_scene_idxs_this_datapoint = np.zeros(self.max_query_length, dtype="int32")
        if self._eval_mode:
            query_scene_srAz_this_datapoint = np.zeros((self.max_query_length, 3), dtype="int32")

        assert self._echo_feat_shape is not None
        context_echoes_mag_this_datapoint = np.zeros((self.max_context_length,
                                                      self._echo_feat_shape[0],
                                                      self._echo_feat_shape[1],
                                                      self._echo_feat_shape[2])).astype("float32")
        gt_queryImpEchoes_mag_this_datapoint = np.zeros((self.max_query_length,
                                                         self._echo_feat_shape[0],
                                                         self._echo_feat_shape[1],
                                                         self._echo_feat_shape[2])).astype("float32")
        if self._eval_mode:
            gt_queryImpEchoes_phase_this_datapoint = np.zeros((self.max_query_length,
                                                               self._echo_feat_shape[0],
                                                               self._echo_feat_shape[1],
                                                               self._echo_feat_shape[2])).astype("float32")

        assert len(context_poses) >= 1, "can't compute relative query pose if there isn't at least 1 valid entry in context"
        ref_pose_for_computing_rel_pose = context_poses[0]

        for context_idx in range(self.max_context_length):
            if context_idx < len(context_poses):
                curr_context_entry_rgb =\
                    self.all_scenes_observations[datapoint_scene][(context_poses[context_idx][0],
                                                                   self._compute_rotation_from_azimuth(context_poses[context_idx][1]))]["rgb"][:, :, :3]
                curr_context_entry_depth = self.all_scenes_observations[datapoint_scene][(context_poses[context_idx][0],
                                                                                          self._compute_rotation_from_azimuth(context_poses[context_idx][1]))]["depth"]
                curr_context_entry_depth = np.expand_dims(curr_context_entry_depth, axis=-1)
                if self.sim_cfg.DEPTH_SENSOR.NORMALIZE_DEPTH:
                    curr_context_entry_depth = self._normalize_depth(curr_context_entry_depth)

                curr_context_entry_view = np.concatenate((curr_context_entry_rgb, curr_context_entry_depth), axis=-1)

                context_views_this_datapoint[context_idx] = curr_context_entry_view

                curr_context_entry_echo_mag, curr_context_entry_echo_phase =\
                    self._compute_spect(scene=datapoint_scene,
                                        azimuth=int(context_poses[context_idx][1]),
                                        receiver_node=int(context_poses[context_idx][0]),
                                        source_node=None,
                                        is_context=True,
                                        )
                context_echoes_mag_this_datapoint[context_idx] = curr_context_entry_echo_mag

                assert len(context_poses[context_idx]) == 2
                current_context_pose = [context_poses[context_idx][0],
                                        context_poses[context_idx][0],
                                        context_poses[context_idx][1]]
                curr_context_entry_pose =\
                    np.array(self._compute_relative_pose(current_pose=current_context_pose,
                                                         ref_pose=ref_pose_for_computing_rel_pose,
                                                         scene_graph=self._all_scenes_graphs_this_split[datapoint_scene],
                                                         )).astype("float32")
                context_poses_this_datapoint[context_idx] = curr_context_entry_pose

                if not self._is_max_context_length_zero:
                    context_mask_this_datapoint[context_idx] = 1

        datapoint["context"] = {}
        datapoint["context"]["views"] = context_views_this_datapoint
        datapoint["context"]["echoes_mag"] = context_echoes_mag_this_datapoint
        datapoint["context"]["poses"] = context_poses_this_datapoint
        datapoint["context"]["mask"] = context_mask_this_datapoint

        for query_idx in range(self.max_query_length):
            if query_idx < len(query_poses):
                curr_query_entry_gtImpEcho_mag, curr_query_entry_gtImpEcho_phase =\
                    self._compute_spect(scene=datapoint_scene,
                                        azimuth=int(query_poses[query_idx][2]),
                                        receiver_node=int(query_poses[query_idx][0]),
                                        source_node=int(query_poses[query_idx][1]),
                                        )
                gt_queryImpEchoes_mag_this_datapoint[query_idx] = curr_query_entry_gtImpEcho_mag
                if self._eval_mode:
                    gt_queryImpEchoes_phase_this_datapoint[query_idx] = curr_query_entry_gtImpEcho_phase

                curr_query_entry_pose =\
                    np.array(self._compute_relative_pose(current_pose=query_poses[query_idx],
                                                         ref_pose=ref_pose_for_computing_rel_pose,
                                                         scene_graph=self._all_scenes_graphs_this_split[datapoint_scene],
                                                         )).astype("float32")
                query_poses_this_datapoint[query_idx] = curr_query_entry_pose

                query_mask_this_datapoint[query_idx] = 1

                assert datapoint_scene in SCENE_NAME_TO_IDX[self.scene_dataset]
                query_scene_idxs_this_datapoint[query_idx] = SCENE_NAME_TO_IDX[self.scene_dataset][datapoint_scene]

                if self._eval_mode:
                    # max_query_length x 3 ... in the order of s, r, Az
                    # s gets assigned
                    query_scene_srAz_this_datapoint[query_idx][0] = int(query_poses[query_idx][1])
                    # az gets assigned
                    query_scene_srAz_this_datapoint[query_idx][2] = int(query_poses[query_idx][2])
                    # r gets assigned
                    query_scene_srAz_this_datapoint[query_idx][1] = int(query_poses[query_idx][0])

        datapoint["query"] = {}

        datapoint["query"]["gt_impEchoes_mag"] = gt_queryImpEchoes_mag_this_datapoint
        if self._eval_mode:
            datapoint["query"]["gt_impEchoes_phase"] = gt_queryImpEchoes_phase_this_datapoint
        datapoint["query"]["poses"] = query_poses_this_datapoint
        datapoint["query"]["mask"] = query_mask_this_datapoint
        datapoint["query"]["scene_idxs"] = query_scene_idxs_this_datapoint
        if self._eval_mode:
            datapoint["query"]["scene_srAzs"] = query_scene_srAz_this_datapoint
        return datapoint

    def _normalize_depth(self, depth):
        """
        normalize depth
        :param depth: unnormalized depth
        :return: normalized depth
        """
        depth = (depth - self.sim_cfg.DEPTH_SENSOR.MIN_DEPTH) / (
            self.sim_cfg.DEPTH_SENSOR.MAX_DEPTH - self.sim_cfg.DEPTH_SENSOR.MIN_DEPTH
        )
        assert np.all(depth <= 1.0) and np.all(depth >= 0.0)

        return depth

    def _compute_relative_pose(self, current_pose=None, ref_pose=None, scene_graph=None,):
        """
        compute relative pose
        :param current_pose: current pose
        :param ref_pose: reference pose
        :param scene_graph: scene graph
        :return: relative pose
        """
        assert isinstance(current_pose, list)
        assert isinstance(ref_pose, list)
        assert len(ref_pose) == 2
        assert len(current_pose) == 3
        assert scene_graph is not None

        ref_position_xyz = np.array(list(scene_graph.nodes[ref_pose[0]]["point"]), dtype=np.float32)
        rotation_world_ref = quat_from_angle_axis(np.deg2rad(self._compute_rotation_from_azimuth(ref_pose[1])),
                                                  np.array([0, 1, 0]))

        agent_position_xyz = np.array(list(scene_graph.nodes[current_pose[0]]["point"]), dtype=np.float32)
        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_ref.inverse(), agent_position_xyz - ref_position_xyz
        )

        audio_source_position_xyz = np.array(list(scene_graph.nodes[current_pose[1]]["point"]), dtype=np.float32)
        audio_source_position_xyz = audio_source_position_xyz - ref_position_xyz

        rotation_world_agent = quat_from_angle_axis(np.deg2rad(self._compute_rotation_from_azimuth(current_pose[2])),
                                                    np.array([0, 1, 0]))
        # next 2 lines compute relative rotation in the counter-clockwise direction, i.e. -z to -x
        # rotation_world_agent.inverse() * rotation_world_ref = rotation_world_agent - rotation_world_ref
        heading_vector = quaternion_rotate_vector(rotation_world_agent.inverse() * rotation_world_ref,
                                                  np.array([0, 0, -1]))
        agent_heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

        return [-agent_position_xyz[2], agent_position_xyz[0], -audio_source_position_xyz[2],
                audio_source_position_xyz[0], agent_heading]

    def _compute_spect(self, scene=None, azimuth=None, receiver_node=None, source_node=None,
                       is_context=False,):
        """
        compute IR spectrogram (spect.)
        :param scene: env name
        :param azimuth: pose azimuth angle
        :param receiver_node: pose receiver node
        :param source_node: pose source node
        :param is_context: flag to tell if spect. for context (echo)
        :return: spect.
        """
        if is_context:
            binaural_rir_file = os.path.join(self.binaural_rir_dir, scene, str(azimuth), f"{receiver_node}_{receiver_node}.wav")
        else:
            binaural_rir_file = os.path.join(self.binaural_rir_dir, scene, str(azimuth), f"{receiver_node}_{source_node}.wav")

        assert os.path.isfile(binaural_rir_file), print(binaural_rir_file)

        try:
            fs_imp, sig_imp = wavfile.read(binaural_rir_file)
            assert fs_imp == self.rir_sampling_rate, "RIR doesn't have sampling frequency of rir_sampling_rate"
        except ValueError:
            sig_imp = np.zeros((self.rir_sampling_rate, 2)).astype("float32")
            fs_imp = self.rir_sampling_rate

        if len(sig_imp) == 0:
            sig_imp = np.zeros((self.rir_sampling_rate, 2)).astype("float32")
            fs_imp = self.rir_sampling_rate

        imp_full_length = np.zeros((self.rir_sampling_rate, 2))
        if sig_imp.shape[0] > 128:
            imp_full_length[: min(sig_imp.shape[0], self.rir_sampling_rate) - 128, :] =\
                sig_imp[128: min(sig_imp.shape[0], self.rir_sampling_rate), :]    # remove the first 127 zero samples
        sig_imp = imp_full_length
        assert fs_imp == self.rir_sampling_rate

        sig_imp = sig_imp.T

        if "BIN_SPECT_MAG_SENSOR" in self.task_cfg.SENSORS:
            fft_windows_l_imp = librosa.stft(np.asfortranarray(sig_imp[0]),
                                             hop_length=self.hop_length,
                                             n_fft=self.n_fft,
                                             win_length=self.win_length if (self.win_length != 0) else None,)
            magnitude_l_imp, phase_l_imp = librosa.magphase(fft_windows_l_imp)
            phase_l_imp = np.angle(phase_l_imp)

            fft_windows_r_imp = librosa.stft(np.asfortranarray(sig_imp[1]),
                                             hop_length=self.hop_length,
                                             n_fft=self.n_fft,
                                             win_length=self.win_length if (self.win_length != 0) else None,)
            magnitude_r_imp, phase_r_imp = librosa.magphase(fft_windows_r_imp)
            phase_r_imp = np.angle(phase_r_imp)

            magnitude_imp = np.stack([magnitude_l_imp, magnitude_r_imp], axis=-1)
            phase_imp = np.stack([phase_l_imp, phase_r_imp], axis=-1)
        else:
            raise ValueError

        magnitude_imp = magnitude_imp.astype("float32")
        phase_imp = phase_imp.astype("float32")

        return magnitude_imp, phase_imp

    def _compute_rotation_from_azimuth(self, azimuth):
        """
        compute rotation angle from azimuth angle
        :param azimuth: azimuth angle
        :return: rotation angle
        """
        # rotation is calculated in the habitat coordinate frame counter-clocwise so -Z is 0 and -X is -90
        return -(azimuth + 0) % 360
