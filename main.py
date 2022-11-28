#!/usr/bin/env python3

import argparse
import logging

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import tensorflow as tf
import torch

from rir_rendering.common.baseline_registry import baseline_registry
from rir_rendering.config.default import get_config
from habitat_audio import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        # required=True,
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--eval-wo-ckpt-path",
        action="store_true",
        default=False,
        help="flag for roomba/nearest-neighbor eval, doesn't need checkpoint path",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        # required=True,
        default='rir_rendering/config/test/uniform_context_sampler.yaml',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Evaluation interval of checkpoints",
    )
    args = parser.parse_args()

    # repo = git.Repo(search_parent_directories=True)
    # logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))

    # run exp
    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type)
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    # torch.set_num_threads(1)

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.run_type == "train":
        trainer.train()
    elif args.run_type == "eval":
        trainer.eval(args.eval_interval, args.prev_ckpt_ind, eval_wo_ckpt_path=args.eval_wo_ckpt_path,)


if __name__ == "__main__":
    main()
