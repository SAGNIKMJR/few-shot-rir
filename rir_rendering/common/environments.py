r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import logging

import habitat
from habitat import Config, Dataset
from rir_rendering.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="ExploreEnv")
class ExploreEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        """
        Environment class
        :param config: environment config
        :param dataset: dataset for environment
        """
        self._rl_config = config.RL
        self._config = config
        self._core_env_config = config.TASK_CONFIG

        self._episode_distance_covered = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        """
        reset environment
        :return: observation after reset
        """
        self._env_step = 0
        observation = super().reset()
        logging.debug(super().current_episode)
        return observation

    def step(self, *args, **kwargs):
        """
        take step in environment
        :param args: arguments
        :param kwargs: keyword arguments
        :return: tuple with observation, reward, done/not-done mask and other episode information after current step
        """
        observation, reward, done, info = super().step(*args, **kwargs)
        self._env_step += 1
        return observation, reward, done, info

    def get_reward_range(self):
        """
        get range of reward
        :return: reward range
        """
        return (
            0,
            float('inf'),
        )   

    def get_reward(self, observations):
        """
        get reward for current step
        :param observations: observations at current step
        :return: reward for current step
        """
        return 0

    def _distance_target(self):
        """
        get distance to target
        :return: distance to target
        """
        return float('inf')

    def _episode_success(self):
        """
        is episode successful or not
        :return: True if episode successful else False
        """
        return False

    def _goal_reached(self):
        """
        is goal reached or not
        :return: True if goal reached else False
        """
        return False

    def get_done(self, observations):
        """
        is episode done or not
        :param observations: observations at current step
        :return: True if episode done else False
        """
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        """
        get episode information
        :param observations: observations at current step
        :return: episode information
        """
        return self.habitat_env.get_metrics()

    def get_current_episode_id(self):
        """
        get current episode ID
        :return: current episode ID
        """
        return self.habitat_env.current_episode.episode_id
