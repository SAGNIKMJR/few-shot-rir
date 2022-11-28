import logging
import random
import os
import numpy as np
import signal
import warnings
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Thread
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    Type,
)

import torch
import torch.nn.functional as F
from torch import multiprocessing as mp
import gym
from gym import spaces

import habitat
from habitat import Config, Env, RLEnv, VectorEnv
from habitat.datasets import make_dataset
from habitat.config import Config
from habitat.core.env import Env, Observations, RLEnv
from habitat.core.logging import logger
from habitat.core.utils import tile_images

from rir_rendering.common.sync_vector_env import SyncVectorEnv

STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
CALL_COMMAND = "call"
COUNT_EPISODES_COMMAND = "count_episodes"


EPISODE_OVER_NAME = "episode_over"
GET_METRICS_NAME = "get_metrics"
OBSERVATION_SPACE_NAME = "observation_space"
ACTION_SPACE_NAME = "action_space"
CURRENT_EPISODE_NAME = "current_episode"
NUMBER_OF_EPISODE_NAME = "number_of_episodes"


def _make_env_fn(
    config: Config, dataset: Optional[habitat.Dataset] = None, rank: int = 0
) -> Env:
    """Constructor for default habitat :ref:`env.Env`.
    :param config: configuration for environment.
    :param dataset: dataset for environment.
    :param rank: rank for setting seed of environment
    :return: :ref:`env.Env` / :ref:`env.RLEnv` object
    """
    habitat_env = Env(config=config, dataset=dataset)
    habitat_env.seed(config.SEED + rank)
    return habitat_env


class VectorEnvCustom:
    r"""Custom vectorized environment from v0.1.6 (master/stable or the one after 0.1.6)which creates multiple processes
    where each process runs its own environment. Main class for parallelization of
    training and evaluation.
    All the environments are synchronized on step and reset methods.
    """

    observation_spaces: List[spaces.Dict]
    number_of_episodes: List[Optional[int]]
    action_spaces: List[spaces.Dict]
    _workers: List[Union[mp.Process, Thread]]
    _is_waiting: bool
    _num_envs: int
    _auto_reset_done: bool
    _mp_ctx: BaseContext
    _connection_read_fns: List[Callable[[], Any]]
    _connection_write_fns: List[Callable[[Any], None]]

    def __init__(
        self,
        make_env_fn: Callable[..., Union[Env, RLEnv]] = _make_env_fn,
        env_fn_args: Sequence[Tuple] = None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
        workers_ignore_signals: bool = False,
    ) -> None:
        """..
        :param make_env_fn: function which creates a single environment. An
            environment can be of type :ref:`env.Env` or :ref:`env.RLEnv`
        :param env_fn_args: tuple of tuple of args to pass to the
            :ref:`_make_env_fn`.
        :param auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        :param multiprocessing_start_method: the multiprocessing method used to
            spawn worker processes. Valid methods are
            :py:`{'spawn', 'forkserver', 'fork'}`; :py:`'forkserver'` is the
            recommended method as it works well with CUDA. If :py:`'fork'` is
            used, the subproccess  must be started before any other GPU useage.
        :param workers_ignore_signals: Whether or not workers will ignore SIGINT and SIGTERM
            and instead will only exit when :ref:`close` is called
        """
        self._is_waiting = False
        self._is_closed = True

        assert (
            env_fn_args is not None and len(env_fn_args) > 0
        ), "number of environments to be created should be greater than 0"

        self._num_envs = len(env_fn_args)

        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        self._auto_reset_done = auto_reset_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args,
            make_env_fn,
            workers_ignore_signals=workers_ignore_signals,
        )

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (OBSERVATION_SPACE_NAME, None)))
        self.observation_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (ACTION_SPACE_NAME, None)))
        self.action_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        self._paused: List[Tuple] = []

    @property
    def num_envs(self):
        r"""number of individual environments."""
        return self._num_envs - len(self._paused)

    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        mask_signals: bool = False,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)

        env = env_fn(*env_fn_args)
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # different step methods for habitat.RLEnv and habitat.Env
                    if isinstance(env, (habitat.RLEnv, gym.Env)):
                        # habitat.RLEnv
                        observations, reward, done, info = env.step(**data)
                        if auto_reset_done and done:
                            observations = env.reset()
                        connection_write_fn(
                            (observations, reward, done, info)
                        )
                    elif isinstance(env, habitat.Env):  # type: ignore
                        # habitat.Env
                        observations = env.step(**data)
                        if auto_reset_done and env.episode_over:
                            observations = env.reset()
                        connection_write_fn(observations)
                    else:
                        raise NotImplementedError

                elif command == RESET_COMMAND:
                    observations = env.reset()
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None:
                        function_args = {}

                    result_or_fn = getattr(env, function_name)

                    if len(function_args) > 0 or callable(result_or_fn):
                        result = result_or_fn(**function_args)
                    else:
                        result = result_or_fn

                    connection_write_fn(result)

                elif command == COUNT_EPISODES_COMMAND:
                    connection_write_fn(len(env.episodes))

                else:
                    raise NotImplementedError(f"Unknown command {command}")

                command, data = connection_read_fn()

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            env.close()

    def _spawn_workers(
        self,
        env_fn_args: Sequence[Tuple],
        make_env_fn: Callable[..., Union[Env, RLEnv]] = _make_env_fn,
        workers_ignore_signals: bool = False,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_connections, worker_connections = zip(
            *[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_envs)]
        )
        self._workers = []
        for worker_conn, parent_conn, env_args in zip(
            worker_connections, parent_connections, env_fn_args
        ):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                    workers_ignore_signals,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(cast(mp.Process, ps))
            ps.daemon = True
            ps.start()
            worker_conn.close()
        return (
            [p.recv for p in parent_connections],
            [p.send for p in parent_connections],
        )

    def current_episodes(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (CURRENT_EPISODE_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def count_episodes(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((COUNT_EPISODES_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def episode_over(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (EPISODE_OVER_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def get_metrics(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (GET_METRICS_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def reset(self):
        r"""Reset all the vectorized environments
        :return: list of outputs from the reset method of envs.
        """
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def reset_at(self, index_env: int):
        r"""Reset in the index_env environment in the vector.
        :param index_env: index of the environment to be reset
        :return: list containing the output of reset method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((RESET_COMMAND, None))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def step_at(self, index_env: int, action: Dict[str, Any]):
        r"""Step in the index_env environment in the vector.
        :param index_env: index of the environment to be stepped into
        :param action: action to be taken
        :return: list containing the output of step method of indexed env.
        """
        self._is_waiting = True
        self._warn_cuda_tensors(action)
        self._connection_write_fns[index_env]((STEP_COMMAND, action))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def async_step(self, data: List[Union[int, str, Dict[str, Any]]]) -> None:
        r"""Asynchronously step in the environments.
        :param data: list of size _num_envs containing keyword arguments to
            pass to :ref:`step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        """
        # Backward compatibility
        if isinstance(data[0], (int, np.integer, str)):
            actions = [{"action": {"action": action}} for action in data]
        else:
            actions = cast(List[Dict[str, Any]], data)

        self._is_waiting = True
        for write_fn, action in zip(self._connection_write_fns, actions):
            self._warn_cuda_tensors(action)
            write_fn((STEP_COMMAND, action))

    def wait_step(self) -> List[Any]:
        r"""Wait until all the asynchronized environments have synchronized."""
        observations = []
        for read_fn in self._connection_read_fns:
            observations.append(read_fn())
        self._is_waiting = False
        return observations

    def step(self, data: List[Union[int, str, Dict[str, Any]]]) -> List[Any]:
        r"""Perform actions in the vectorized environments.
        :param data: list of size _num_envs containing keyword arguments to
            pass to :ref:`step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        :return: list of outputs from the step method of envs.
        """
        self.async_step(data)
        return self.wait_step()

    def close(self) -> None:
        if self._is_closed:
            return

        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()

        for write_fn in self._connection_write_fns:
            write_fn((CLOSE_COMMAND, None))

        for _, _, write_fn, _ in self._paused:
            write_fn((CLOSE_COMMAND, None))

        for process in self._workers:
            process.join()

        for _, _, _, process in self._paused:
            process.join()

        self._is_closed = True

    def pause_at(self, index: int) -> None:
        r"""Pauses computation on this env without destroying the env.
        :param index: which env to pause. All indexes after this one will be
            shifted down by one.
        This is useful for not needing to call steps on all environments when
        only some are active (for example during the last episodes of running
        eval episodes).
        """
        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()
        read_fn = self._connection_read_fns.pop(index)
        write_fn = self._connection_write_fns.pop(index)
        worker = self._workers.pop(index)
        self._paused.append((index, read_fn, write_fn, worker))

    def resume_all(self) -> None:
        r"""Resumes any paused envs."""
        for index, read_fn, write_fn, worker in reversed(self._paused):
            self._connection_read_fns.insert(index, read_fn)
            self._connection_write_fns.insert(index, write_fn)
            self._workers.insert(index, worker)
        self._paused = []

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        r"""Calls a function or retrieves a property/member variable (which is passed by name)
        on the selected env and returns the result.
        :param index: which env to call the function on.
        :param function_name: the name of the function to call or property to retrieve on the env.
        :param function_args: optional function args.
        :return: result of calling the function.
        """
        self._is_waiting = True
        self._connection_write_fns[index](
            (CALL_COMMAND, (function_name, function_args))
        )
        result = self._connection_read_fns[index]()
        self._is_waiting = False
        return result

    def call(
        self,
        function_names: List[str],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        r"""Calls a list of functions (which are passed by name) on the
        corresponding env (by index).
        :param function_names: the name of the functions to call on the envs.
        :param function_args_list: list of function args for each function. If
            provided, :py:`len(function_args_list)` should be as long as
            :py:`len(function_names)`.
        :return: result of calling the function.
        """
        self._is_waiting = True
        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(
            self._connection_write_fns, func_args
        ):
            write_fn((CALL_COMMAND, func_args_on))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def render(
        self, mode: str = "human", *args, **kwargs
    ) -> Union[np.ndarray, None]:
        r"""Render observations from all environments in a tiled image."""
        for write_fn in self._connection_write_fns:
            write_fn((RENDER_COMMAND, (args, {"mode": "rgb", **kwargs})))
        images = [read_fn() for read_fn in self._connection_read_fns]
        tile = tile_images(images)
        if mode == "human":
            from habitat.core.utils import try_cv2_import

            cv2 = try_cv2_import()

            cv2.imshow("vecenv", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {"forkserver", "spawn", "fork"}

    def _warn_cuda_tensors(
        self, action: Dict[str, Any], prefix: Optional[str] = None
    ):
        if torch is None:
            return

        for k, v in action.items():
            if isinstance(v, dict):
                subk = f"{prefix}.{k}" if prefix is not None else k
                self._warn_cuda_tensors(v, prefix=subk)
            elif torch.is_tensor(v) and v.device.type == "cuda":
                subk = f"{prefix}.{k}" if prefix is not None else k
                warnings.warn(
                    "Action with key {} is a CUDA tensor."
                    "  This will result in a CUDA context in the subproccess worker."
                    "  Using CPU tensors instead is recommended.".format(subk)
                )

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def construct_envs(
    config: Config, env_class: Type[Union[Env, RLEnv]], workers_ignore_signals: bool = False,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
        env_class: class type of the envs to be created
    Returns:
        VectorEnv object created according to specification.
    """

    num_processes = config.NUM_PROCESSES
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if (config.EPS_SCENES != []) and config.TASK_CONFIG.DATASET.SPLIT[:5] == "train":
        scenes = config.EPS_SCENES

    if len(scenes) > 0:
        if config.TASK_CONFIG.DATASET.SPLIT[:5] == "train":
            random.shuffle(scenes)
        assert len(scenes) >= num_processes, (
            "reduce the number of task_config.SIMULATOR.SOUNDS_IN_SPLIT processes as there "
            "aren't enough number of scenes"
        )

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        # set only for ddppo
        if workers_ignore_signals:
            proc_config = config.clone()
            proc_config.defrost()

            task_config = proc_config.TASK_CONFIG
            proc_config.SEED = proc_config.SEED + i
            task_config.SIMULATOR.SEED = proc_config.SEED
        else:
            task_config = config.TASK_CONFIG.clone()
            task_config.defrost()

        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]
            logging.debug('All scenes: {}'.format(','.join(scene_splits[i])))

        # overwrite the task config with top-level config file
        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        if not workers_ignore_signals:
            task_config.freeze()

            config.defrost()
            config.TASK_CONFIG = task_config
            config.freeze()
            configs.append(config.clone())
        else:
            proc_config.freeze()
            configs.append(proc_config)

    if not workers_ignore_signals:
        # use VectorEnv for the best performance and ThreadedVectorEnv for debugging
        if config.USE_SYNC_VECENV:
            env_launcher = SyncVectorEnv
            logging.info('Using SyncVectorEnv')
        elif config.USE_VECENV:
            env_launcher = habitat.VectorEnv
            logging.info('Using VectorEnv')
        else:
            env_launcher = habitat.ThreadedVectorEnv
            logging.info('Using ThreadedVectorEnv')
        envs = env_launcher(
            make_env_fn=make_env_fn,
            env_fn_args=tuple(
                tuple(zip(configs, env_classes, range(num_processes)))
            ),
        )
    else:
        envs = VectorEnvCustom(
            make_env_fn=make_env_fn_custom,
            env_fn_args=tuple(
                tuple(zip(configs, env_classes))
            ),
        )
    return envs


def make_env_fn(
    config: Config, env_class: Type[Union[Env, RLEnv]], rank: int
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.
    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
        rank: rank of env to be created (for seeding).
    Returns:
        env object created according to specification.
    """
    if not config.USE_SYNC_VECENV:
        level = logging.DEBUG if config.DEBUG else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S")
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    # filter out certain scenes (+ episode IDs) during eval
    if (config.EPS_SCENES != []) and (config.TASK_CONFIG.DATASET.SPLIT[:5] != "train"):
        dataset.filter_by_scenes(config.EPS_SCENES)
    elif (config.EPS_SCENES_N_IDS != []) and (config.TASK_CONFIG.DATASET.SPLIT[:5] != "train"):
        dataset.filter_by_scenes_n_ids(config.EPS_SCENES_N_IDS)
    env = env_class(config=config, dataset=dataset)
    env.seed(rank)
    return env


def make_env_fn_custom(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.
    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
    Returns:
        env object created according to specification.
    """
    if not config.USE_SYNC_VECENV:
        level = logging.DEBUG if config.DEBUG else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S")
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    # filter out certain scenes during eval
    if (config.EPS_SCENES != []) and (config.TASK_CONFIG.DATASET.SPLIT[:5] != "train"):
        dataset.filter_by_scenes(config.EPS_SCENES)
    env = env_class(config=config, dataset=dataset)
    env.seed(config.SEED)
    return env
