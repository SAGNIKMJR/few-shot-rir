import os
import logging
import random
import pickle
from PIL import Image
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader

from habitat import logger

from rir_rendering.common.base_trainer import BaseRLTrainer
from rir_rendering.common.baseline_registry import baseline_registry
from rir_rendering.common.env_utils import construct_envs
from rir_rendering.common.environments import get_env_class
from rir_rendering.common.tensorboard_utils import TensorboardWriter
from rir_rendering.common.losses import compute_spect_losses, compute_spect_energy_decay_losses
from rir_rendering.common.eval_metrics import compute_spect_metrics
from rir_rendering.uniform_context_sampler.policy import UniformContextSamplerPolicy
from rir_rendering.uniform_context_sampler.uniform_context_sampler import UniformContextSampler
from rir_rendering.datasets.dataset import UniformContextSamplerDataset
from habitat_audio.utils import load_points_data


SCENE_IDX_TO_NAME = {
    "mp3d":
        {0: 'sT4fr6TAbpF', 1: 'E9uDoFAP3SH', 2: 'VzqfbhrpDEA', 3: 'kEZ7cmS4wCh', 4: '29hnd4uzFmX', 5: 'ac26ZMwG7aT',
         6: 's8pcmisQ38h', 7: 'rPc6DW4iMge', 8: 'EDJbREhghzL', 9: 'mJXqzFtmKg4', 10: 'B6ByNegPMKs', 11: 'JeFG25nYj2p',
         12: '82sE5b5pLXE', 13: 'D7N2EKCX4Sj', 14: '7y3sRwLe3Va', 15: '5LpN3gDmAk7', 16: 'gTV8FGcVJC9', 17: 'ur6pFq6Qu1A',
         18: 'qoiz87JEwZ2', 19: 'PuKPg4mmafe', 20: 'VLzqgDo317F', 21: 'aayBHfsNo7d', 22: 'JmbYfDe2QKZ', 23: 'XcA2TqTSSAj',
         24: '8WUmhLawc2A', 25: 'sKLMLpTHeUy', 26: 'r47D5H71a5s', 27: 'Uxmj2M2itWa', 28: 'Pm6F8kyY3z2', 29: 'p5wJjkQkbXX',
         30: '759xd9YjKW5', 31: 'JF19kD82Mey', 32: 'V2XKFyX4ASd', 33: '1LXtFkjw3qL', 34: '17DRP5sb8fy', 35: '5q7pvUzZiYa',
         36: 'VVfe2KiqLaN', 37: 'Vvot9Ly1tCj', 38: 'ULsKaCPVFJR', 39: 'D7G3Y4RVNrH', 40: 'uNb9QFRL6hY', 41: 'ZMojNkEp431',
         42: '2n8kARJN3HM', 43: 'vyrNrziPKCB', 44: 'e9zR4mvMWw7', 45: 'r1Q1Z4BcV1o', 46: 'PX4nDJXEHrG', 47: 'YmJkqBEsHnH',
         48: 'b8cTxDM8gDG', 49: 'GdvgFV5R1Z5', 50: 'pRbA3pwrgk9', 51: 'jh4fc5c5qoQ', 52: '1pXnuDYAj8r', 53: 'S9hNv5qa7GM',
         54: 'VFuaQ6m2Qom', 55: 'cV4RVeZvu5T', 56: 'SN83YJsR3w2', 57: '2azQ1b91cZZ', 58: '5ZKStnWn8Zo', 59: '8194nk5LbLH',
         60: 'ARNzJeq3xxb', 61: 'EU6Fwq7SyZv', 62: 'QUCTc6BB5sX', 63: 'TbHJrupSAjP', 64: 'UwV83HsGsw3', 65: 'Vt2qJdWjCF2',
         66: 'WYY7iVyf5p8', 67: 'X7HyMhZNoso', 68: 'YFuZgdQ5vWj', 69: 'Z6MFQCViBuw', 70: 'fzynW3qQPVF', 71: 'gYvKGZ5eRqb',
         72: 'gxdoqLR6rwA', 73: 'jtcxE69GiFV', 74: 'oLBMNvg9in8', 75: 'pLe4wQe7qrG', 76: 'pa4otMbVnkk', 77: 'q9vSo1VnCiC',
         78: 'rqfALeAoiTq', 79: 'wc2JMjhGNzB', 80: 'x8F5xyUWy9e', 81: 'yqstnuAEVhm', 82: 'zsNo4HB9uLZ'},
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

LOSS_LIKE_METRICS = ["stft_l1_distance"]


@baseline_registry.register_trainer(name="uniform_context_sampler")
class UniformContextSamplerTrainer(BaseRLTrainer):
    """Trainer class for training IR predictor using uniform sampling of context in a supervised fashion
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self._n_available_gpus = None

    def _setup_uniform_context_sampler_agent(self, use_data_parallel=False) -> None:
        """
        Sets up agent for IR prediction using uniform sampling of context.
        :param use_data_parallel: flag to tell whether to use data_parallel or not
        :return: None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = UniformContextSamplerPolicy(
            cfg=self.config,
        )

        self.actor_critic.to(self.device)

        self._n_available_gpus = torch.cuda.device_count()
        self._use_data_parallel = use_data_parallel
        if ((self._n_available_gpus > 0) or use_data_parallel):
            print("Using", torch.cuda.device_count(), "GPUs!")

            self.actor_critic = nn.DataParallel(self.actor_critic, device_ids=list(range(self._n_available_gpus)),
                                                output_device=0)

        self.agent = UniformContextSampler(
            actor_critic=self.actor_critic,
        )

    def save_checkpoint(self, file_name: str,) -> None:
        """
        Save checkpoint with specified name.
        :param file_name: file name for checkpoint
        :return: None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """
        Load checkpoint of specified path as a dict.
        :param checkpoint_path: path of target checkpoint
        :param args: additional positional args
        :param kwargs: additional keyword args
        :return: dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def get_dataloaders(self, eval_mode=False, ckpt_rootdir_path=None):
        """
        build datasets and dataloaders
        :param eval_mode: flag to tell if model is in eval mode or not
        :param ckpt_rootdir_path: run directory path
        :return:
            dataloaders: PyTorch dataloaders for training and validation
            dataset_sizes: sizes of train and val datasets
        """
        uniform_context_sampler_cfg = self.config.UniformContextSampler
        task_cfg = self.config.TASK_CONFIG
        sim_cfg = task_cfg.SIMULATOR
        audio_cfg = sim_cfg.AUDIO

        scene_dataset = sim_cfg.SCENE_DATASET

        scene_observations_dir = os.path.join(sim_cfg.RENDERED_OBSERVATIONS, scene_dataset)
        assert os.path.isdir(scene_observations_dir)
        all_scenes_observations = dict()
        if os.path.isfile(os.path.join(scene_observations_dir, "scene_observations.pkl")):
            all_scenes_observations_file_path = os.path.join(scene_observations_dir, "scene_observations.pkl")
            with open(all_scenes_observations_file_path, "rb") as fi:
                all_scenes_observations = pickle.load(fi)
        else:
            all_scenes_lst = []
            for split_type in SCENE_SPLITS[scene_dataset]:
                all_scenes_lst += SCENE_SPLITS[scene_dataset][split_type]

            for scene in all_scenes_lst:
                scene_observations_file_path = os.path.join(scene_observations_dir, f"{scene}.pkl")
                with open(scene_observations_file_path, "rb") as fi:
                    all_scenes_observations[scene] = pickle.load(fi)

        scene_splits = {}
        if not eval_mode:
            scene_splits["train"] = SCENE_SPLITS[sim_cfg.SCENE_DATASET]["train"]
        scene_splits["seen_eval"] = SCENE_SPLITS[sim_cfg.SCENE_DATASET]["train"]
        scene_splits["unseen_eval"] = SCENE_SPLITS[sim_cfg.SCENE_DATASET]["unseen_eval"]

        datasets = dict()
        dataloaders = dict()
        dataset_sizes = dict()
        for split in scene_splits:
            scenes = scene_splits[split]
            all_scenes_graphs_this_split = dict()
            for scene in scenes:
                _, graph = load_points_data(
                    os.path.join(audio_cfg.META_DIR, scene),
                    audio_cfg.GRAPH_FILE,
                    transform=True,
                    scene_dataset=sim_cfg.SCENE_DATASET)
                all_scenes_graphs_this_split[scene] = graph

            datasets[split] = UniformContextSamplerDataset(
                split=split,
                all_scenes_graphs_this_split=all_scenes_graphs_this_split,
                cfg=self.config,
                all_scenes_observations=all_scenes_observations,
                eval_mode=eval_mode,
                ckpt_rootdir_path=ckpt_rootdir_path,
            )

            dataloaders[split] = DataLoader(dataset=datasets[split],
                                            batch_size=uniform_context_sampler_cfg.batch_size,
                                            shuffle=(split == 'train'),
                                            pin_memory=True,
                                            num_workers=uniform_context_sampler_cfg.num_workers,
                                            )

            dataset_sizes[split] = len(datasets[split])
            print('{} has {} samples'.format(split.upper(), dataset_sizes[split]))

        if eval_mode:
            return datasets, dataloaders, dataset_sizes
        else:
            return dataloaders, dataset_sizes

    def _optimize_loss(self, loss, optimizer):
        """
        do backward pass and update model parameters
        :param loss: differentiable loss
        :param optimizer: optimizer before update
        :return: optimizer after update
        """
        optimizer.zero_grad()
        if self.config.UniformContextSampler.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.config.UniformContextSampler.max_grad_norm
            )
        loss.backward()
        optimizer.step()

        return optimizer

    def train(self) -> None:
        """Main method for training IR prediction using uniform sampling of context using supervised learning.
        :return: None
        """
        uniform_context_sampler_cfg = self.config.UniformContextSampler
        audio_cfg = self.config.TASK_CONFIG.SIMULATOR.AUDIO

        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_uniform_context_sampler_agent()
        assert self._n_available_gpus is not None

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()),
                                     lr=uniform_context_sampler_cfg.lr,
                                     betas=tuple(uniform_context_sampler_cfg.betas),
                                     eps=uniform_context_sampler_cfg.eps,)

        # build datasets and dataloaders
        dataloaders, dataset_sizes = self.get_dataloaders()

        all_metric_types = uniform_context_sampler_cfg.EvalMetrics.types
        metric_type_for_ckpt_dump = uniform_context_sampler_cfg.EvalMetrics.type_for_ckpt_dump
        loss_like_metric = (metric_type_for_ckpt_dump in LOSS_LIKE_METRICS)

        assert metric_type_for_ckpt_dump in all_metric_types
        if loss_like_metric:
            best_seen_eval_metric_for_ckpt_dump = float('inf')
            best_unseen_eval_metric_for_ckpt_dump = float('inf')
        else:
            raise ValueError

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for epoch in range(uniform_context_sampler_cfg.num_epochs):
                logging.info('-' * 10)
                logging.info('Epoch {}/{}'.format(epoch + 1, uniform_context_sampler_cfg.num_epochs))

                for split in dataloaders.keys():
                    # set forward pass mode
                    if split == "train":
                        self.actor_critic.train()
                    else:
                        self.actor_critic.eval()

                    eval_metrics_epoch = {}
                    if uniform_context_sampler_cfg.use_spect_energy_decay_loss:
                        spect_energy_decay_loss_epoch = 0.

                    for i, data in enumerate(tqdm(dataloaders[split])):
                        context_views = data[0].to(self.device)
                        context_echoes_mag = data[1].to(self.device)
                        context_poses = data[2].to(self.device)
                        context_mask = data[3].to(self.device)
                        gt_queryImpEchoes_mag = data[4].to(self.device)
                        query_poses = data[5].to(self.device)
                        query_mask = data[6].to(self.device)
                        query_scene_idxs = data[7].to(self.device)

                        B = context_echoes_mag.size(0)

                        obs_batch = {"context_views": context_views,
                                     "context_echoes": context_echoes_mag,
                                     "context_poses": context_poses,
                                     "context_mask": context_mask,
                                     "query_poses": query_poses,
                                     "query_mask": query_mask,
                                     "query_scene_idxs": query_scene_idxs,}

                        if split == "train":
                            preds = self.actor_critic(obs_batch)
                        else:
                            with torch.no_grad():
                                preds = self.actor_critic(obs_batch)

                        if self.config.TASK_CONFIG.ENVIRONMENT.MAX_QUERY_LENGTH == 1:
                            assert torch.all(query_mask == 1).item(),\
                                "set config.TASK_CONFIG.ENVIRONMENT.MAX_QUERY_LENGTH to 1 for eval"

                        if uniform_context_sampler_cfg.use_spect_energy_decay_loss:
                            if uniform_context_sampler_cfg.predict_in_logspace:
                                if uniform_context_sampler_cfg.log_instead_of_log1p_in_logspace:
                                    pred_spect_mag = torch.exp(preds.view(-1, *preds.size()[2:]))\
                                                     - uniform_context_sampler_cfg.log_gt_eps
                                else:
                                    pred_spect_mag = torch.exp(preds.view(-1, *preds.size()[2:])) - 1
                            else:
                                pred_spect_mag = preds.view(-1, *preds.size()[2:])

                            gt_spect_mag = gt_queryImpEchoes_mag.view(-1, *gt_queryImpEchoes_mag.size()[2:])

                        if split == "train":
                            loss = compute_spect_losses(
                                loss_types=uniform_context_sampler_cfg.TrainLosses.types,
                                loss_weights=uniform_context_sampler_cfg.TrainLosses.weights,
                                gt_spect=gt_queryImpEchoes_mag.view(-1, *gt_queryImpEchoes_mag.size()[2:]),
                                pred_spect=preds.view(-1, *preds.size()[2:]),
                                mask=query_mask.view(-1),
                                logspace=uniform_context_sampler_cfg.predict_in_logspace,
                                log1p_gt=uniform_context_sampler_cfg.log1p_gt,
                                log_instead_of_log1p_in_logspace=uniform_context_sampler_cfg.log_instead_of_log1p_in_logspace,
                                log_gt=uniform_context_sampler_cfg.log_gt,
                                log_gt_eps=uniform_context_sampler_cfg.log_gt_eps,
                            )

                            if uniform_context_sampler_cfg.use_spect_energy_decay_loss:
                                spect_energy_decay_loss = compute_spect_energy_decay_losses(
                                    loss_type=uniform_context_sampler_cfg.spectEnergyDecayLoss.type,
                                    loss_weight=uniform_context_sampler_cfg.spectEnergyDecayLoss.weight,
                                    gts=gt_spect_mag,
                                    preds=pred_spect_mag,
                                    mask=query_mask.view(-1),
                                    slice_till_direct_signal=uniform_context_sampler_cfg.spectEnergyDecayLoss.slice_till_direct_signal,
                                    direct_signal_len_in_ms=uniform_context_sampler_cfg.spectEnergyDecayLoss.direct_signal_len_in_ms,
                                    dont_collapse_across_freq_dim=uniform_context_sampler_cfg.spectEnergyDecayLoss.dont_collapse_across_freq_dim,
                                    sr=audio_cfg.RIR_SAMPLING_RATE,
                                    hop_length=audio_cfg.HOP_LENGTH,
                                    win_length=audio_cfg.WIN_LENGTH,
                                )

                                loss = loss + spect_energy_decay_loss

                            optimizer = self._optimize_loss(loss, optimizer)

                        # preds, gt_queryImpEchoes_mag: B x max_query_length x H x W x C
                        # query_mask: B x max_query_length
                        # view function called to flatten B x max_query_length -> (B * max_query_length)
                        if uniform_context_sampler_cfg.predict_in_logspace:
                            if uniform_context_sampler_cfg.log_instead_of_log1p_in_logspace:
                                pred_spect_mag = torch.exp(preds.view(-1, *preds.size()[2:]).detach())\
                                                 - uniform_context_sampler_cfg.log_gt_eps
                            else:
                                pred_spect_mag = torch.exp(preds.view(-1, *preds.size()[2:]).detach()) - 1
                        else:
                            pred_spect_mag = preds.view(-1, *preds.size()[2:]).detach()

                        eval_metrics_batch = compute_spect_metrics(
                            metric_types=uniform_context_sampler_cfg.EvalMetrics.types,
                            gt_spect_mag=gt_queryImpEchoes_mag.view(-1, *gt_queryImpEchoes_mag.size()[2:]),
                            pred_spect_mag=pred_spect_mag,
                            mask=query_mask.view(-1),
                            eval_mode=False,
                        )

                        if uniform_context_sampler_cfg.use_spect_energy_decay_loss:
                            spect_energy_decay_loss = compute_spect_energy_decay_losses(
                                loss_type="l1_loss",
                                loss_weight=1.0,
                                gts=gt_queryImpEchoes_mag.view(-1, *gt_queryImpEchoes_mag.size()[2:]),
                                preds=pred_spect_mag,
                                mask=query_mask.view(-1),
                                slice_till_direct_signal=uniform_context_sampler_cfg.spectEnergyDecayLoss.slice_till_direct_signal,
                                direct_signal_len_in_ms=uniform_context_sampler_cfg.spectEnergyDecayLoss.direct_signal_len_in_ms,
                                dont_collapse_across_freq_dim=uniform_context_sampler_cfg.spectEnergyDecayLoss.dont_collapse_across_freq_dim,
                                sr=audio_cfg.RIR_SAMPLING_RATE,
                                hop_length=audio_cfg.HOP_LENGTH,
                                win_length=audio_cfg.WIN_LENGTH,
                            )
                            spect_energy_decay_loss_epoch += (B * spect_energy_decay_loss)

                        for metric_type in all_metric_types:
                            if metric_type not in eval_metrics_epoch:
                                assert metric_type in eval_metrics_batch
                                eval_metrics_epoch[metric_type] = (eval_metrics_batch[metric_type].item() * B)
                            else:
                                eval_metrics_epoch[metric_type] += (eval_metrics_batch[metric_type].item() * B)

                    if uniform_context_sampler_cfg.use_spect_energy_decay_loss:
                        eval_metrics_epoch["spect_energy_decay_loss"] = spect_energy_decay_loss_epoch

                    for metric_type in eval_metrics_epoch.keys():
                        eval_metrics_epoch[metric_type] /= dataset_sizes[split]

                        writer.add_scalar('{}/{}'.format(metric_type, split),
                                          eval_metrics_epoch[metric_type],
                                          epoch)

                        logging.info('{} -- {}: {:.4f}'.format(split.upper(),
                                                               metric_type,
                                                               eval_metrics_epoch[metric_type],
                                                               ))

                    if split == "seen_eval":
                        if eval_metrics_epoch[metric_type_for_ckpt_dump] < best_seen_eval_metric_for_ckpt_dump:
                            best_seen_eval_metric_for_ckpt_dump = eval_metrics_epoch[metric_type_for_ckpt_dump]
                            self.save_checkpoint(f"seen_eval_best_ckpt.pth")
                    elif split == "unseen_eval":
                        if eval_metrics_epoch[metric_type_for_ckpt_dump] < best_unseen_eval_metric_for_ckpt_dump:
                            best_unseen_eval_metric_for_ckpt_dump = eval_metrics_epoch[metric_type_for_ckpt_dump]
                            self.save_checkpoint(f"unseen_eval_best_ckpt.pth")

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ) -> Dict:
        """
        evaluate checkpoint
        :param checkpoint_path:  checkpoint path
        :param writer: tb writer
        :param checkpoint_index: checkpoint index
        :return: None
        """
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        # map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        config = self.config.clone()
        logger.info(f"config: {config}")

        uniform_context_sampler_cfg = config.UniformContextSampler
        audio_cfg = config.TASK_CONFIG.SIMULATOR.AUDIO

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self._setup_uniform_context_sampler_agent(use_data_parallel=config.EVAL.DATA_PARALLEL_TRAINING,)
        assert self._n_available_gpus is not None

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        self.actor_critic.eval()
        self.agent.eval()
        self.agent.actor_critic.eval()

        datasets, dataloaders, dataset_sizes = self.get_dataloaders(eval_mode=True, ckpt_rootdir_path=config.MODEL_DIR)

        all_metric_types = uniform_context_sampler_cfg.EvalMetrics.types

        for split in dataloaders.keys():
            audio_waveforms_dump_dir = None
            if uniform_context_sampler_cfg.dump_audio_waveforms:
                audio_waveforms_dump_dir = os.path.join(config.MODEL_DIR, "audio_waveforms", split)
                if not os.path.isdir(audio_waveforms_dump_dir):
                    os.makedirs(audio_waveforms_dump_dir)

            eval_metrics = {}
            for i, data in enumerate(tqdm(dataloaders[split])):
                context_views = data[0].to(self.device)
                context_echoes_mag = data[1].to(self.device)
                context_poses = data[2].to(self.device)
                context_mask = data[3].to(self.device)
                gt_queryImpEchoes_mag = data[4].to(self.device)
                gt_queryImpEchoes_phase = data[5].to(self.device)
                query_poses = data[6].to(self.device)
                query_mask = data[7].to(self.device)
                query_scene_idxs = data[8].to(self.device)
                query_scene_srAzs = data[9].to(self.device)

                obs_batch = {"context_views": context_views,
                             "context_echoes": context_echoes_mag,
                             "context_poses": context_poses,
                             "context_mask": context_mask,
                             "query_poses": query_poses,
                             "query_mask": query_mask,
                             "query_scene_idxs": query_scene_idxs,
                             }

                with torch.no_grad():
                    preds = self.actor_critic(obs_batch,)

                assert torch.all(query_mask == 1).item(),\
                    "set config.TASK_CONFIG.ENVIRONMENT.MAX_QUERY_LENGTH to 1 for eval"

                assert len(query_scene_idxs.size()) == 2
                assert len(query_scene_srAzs.size()) == 3
                assert query_scene_idxs.size(0) == query_scene_srAzs.size(0)

                # query_scene_idxs: B x max_query_length ... same scene_idx across row
                query_scene_idxs_for_metrics = query_scene_idxs.view(-1)
                lst_query_scene_idxs_for_metrics = query_scene_idxs_for_metrics.detach().cpu().numpy().tolist()

                query_scene_srAzs_for_metrics = query_scene_srAzs.view(-1, query_scene_srAzs.size(2))
                lst_query_scene_srAzs_for_metrics = query_scene_srAzs_for_metrics.detach().cpu().numpy().tolist()

                assert len(lst_query_scene_srAzs_for_metrics) == len(lst_query_scene_idxs_for_metrics)

                eval_scenes_this_batch = []
                eval_srAzs_this_batch = []
                for j, query_scene_idx in enumerate(lst_query_scene_idxs_for_metrics):
                    assert query_scene_idx in SCENE_IDX_TO_NAME[config.TASK_CONFIG.SIMULATOR.SCENE_DATASET]
                    eval_scenes_this_batch.append(SCENE_IDX_TO_NAME[config.TASK_CONFIG.SIMULATOR.SCENE_DATASET][query_scene_idx])

                    assert j < len(lst_query_scene_srAzs_for_metrics)
                    eval_srAzs_this_batch.append(tuple(lst_query_scene_srAzs_for_metrics[j]))

                # preds, gt_queryImpEchoes_mag: B x max_query_length x H x W x C
                # query_mask: B x max_query_length
                # view function called to flatten B x max_query_length -> (B * max_query_length)
                if uniform_context_sampler_cfg.predict_in_logspace:
                    if uniform_context_sampler_cfg.log_instead_of_log1p_in_logspace:
                        pred_spect_mag = torch.exp(preds.view(-1, *preds.size()[2:]).detach())\
                                         - uniform_context_sampler_cfg.log_gt_eps
                    else:
                        pred_spect_mag = torch.exp(preds.view(-1, *preds.size()[2:]).detach()) - 1
                else:
                    pred_spect_mag = preds.view(-1, *preds.size()[2:]).detach()

                eval_metrics_batch = compute_spect_metrics(
                    metric_types=uniform_context_sampler_cfg.EvalMetrics.types,
                    gt_spect_mag=gt_queryImpEchoes_mag.view(-1, *gt_queryImpEchoes_mag.size()[2:]),
                    gt_spect_phase=gt_queryImpEchoes_phase.view(-1, *gt_queryImpEchoes_phase.size()[2:]),
                    pred_spect_mag=pred_spect_mag,
                    eval_mode=True,
                    fs=config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                    hop_length=config.TASK_CONFIG.SIMULATOR.AUDIO.HOP_LENGTH,
                    n_fft=config.TASK_CONFIG.SIMULATOR.AUDIO.N_FFT,
                    win_length=config.TASK_CONFIG.SIMULATOR.AUDIO.WIN_LENGTH,
                    dump_audio_waveforms=uniform_context_sampler_cfg.dump_audio_waveforms,
                    audio_waveforms_dump_dir=audio_waveforms_dump_dir,
                    start_datapoint_idx_for_batch=(i * uniform_context_sampler_cfg.batch_size),
                    eval_scenes_this_batch=eval_scenes_this_batch,
                    eval_srAzs_this_batch=eval_srAzs_this_batch,
                    use_gl=uniform_context_sampler_cfg.use_gl,
                    use_gl_for_gt=uniform_context_sampler_cfg.use_gl_for_gt,
                    use_rand_phase=uniform_context_sampler_cfg.use_rand_phase,
                    use_rand_phase_for_gt=uniform_context_sampler_cfg.use_rand_phase_for_gt,
                )

            for metric_type in all_metric_types:
                if metric_type not in eval_metrics:
                    assert metric_type in eval_metrics_batch
                    eval_metrics[metric_type] = eval_metrics_batch[metric_type]
                else:
                    eval_metrics[metric_type] += eval_metrics_batch[metric_type]

            for metric_type in all_metric_types:
                writer.add_scalar('{}/{}/mean'.format(metric_type, split),
                                  np.mean(eval_metrics[metric_type]),
                                  0)

                writer.add_scalar('{}/{}/std'.format(metric_type, split),
                                  np.std(eval_metrics[metric_type]),
                                  0)

                logger.info(f"{split.upper()} -- {metric_type}: "
                            f" mean -- {np.mean(eval_metrics[metric_type]):.4f}, std -- {np.std(eval_metrics[metric_type]):.4f}")

            with open(os.path.join(config.MODEL_DIR, f"{split}_{dataset_sizes[split]}datapoints_metrics.pkl"), "wb") as fo:
                pickle.dump(eval_metrics, fo, protocol=pickle.HIGHEST_PROTOCOL)

            datasets[split].dump_eval_pkls()
