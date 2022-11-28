from typing import List
from collections import defaultdict
import logging
import pickle
import os

import librosa
import numpy as np
import networkx as nx
from scipy.io import wavfile
from scipy.signal import fftconvolve

from habitat.core.registry import registry
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.simulator import (Config, AgentState, ShortestPathPoint)
from habitat_audio.utils import load_points_data, _to_tensor


class DummySimulator:
    def __init__(self):
        self.position = None
        self.rotation = None
        self._sim_obs = None

    def seed(self, seed):
        pass

    def set_agent_state(self, position, rotation):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation

    def get_agent_state(self):
        class State:
            def __init__(self, position, rotation):
                self.position = position
                self.rotation = rotation

        return State(self.position, self.rotation)

    def set_sensor_observations(self, sim_obs):
        self._sim_obs = sim_obs

    def get_sensor_observations(self):
        return self._sim_obs

    def close(self):
        pass


@registry.register_simulator()
class HabitatSimAudioEnabledTrain(HabitatSim):
    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        r"""Changes made to simulator wrapper over habitat-sim

        This simulator allows the agent to be moved to location specified in the
        Args:
            config: configuration for initializing the simulator.
        """
        super().__init__(config)

        assert self.config.SCENE_DATASET in ["mp3d"], "SCENE_DATASET needs to be in ['mp3d']"
        self.temp_scene_dataset = self.config.SCENE_DATASET
        self._receiver_position_index = None
        self._rotation_angle = None
        self._frame_cache = defaultdict(dict)
        self._is_episode_active = None
        self._previous_step_collided = None
        self._position_to_index_mapping = dict()
        self.points, self.graph = load_points_data(self.meta_dir, self.config.AUDIO.GRAPH_FILE,
                                                   scene_dataset=self.temp_scene_dataset)
        for node in self.graph.nodes():
            self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        logging.info('Current scene: {}'.format(self.current_scene_name,))

        if self.config.USE_RENDERED_OBSERVATIONS:
            logging.info('Loaded the rendered observations for all scenes')
            with open(self.current_scene_observation_file, 'rb') as fo:
                self._frame_cache = pickle.load(fo)
            self._sim.close()
            del self._sim
            self._sim = DummySimulator()

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        r"""
        get current agent state
        :param agent_id: agent ID
        :return: agent state
        """
        if not self.config.USE_RENDERED_OBSERVATIONS:
            agent_state = super().get_agent_state(agent_id)
        else:
            agent_state = self._sim.get_agent_state()

        return agent_state

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        r"""
        set agent's state when not using pre-rendered observations
        :param position: 3D position of the agent
        :param rotation: rotation angle of the agent
        :param agent_id: agent ID
        :param reset_sensors: reset sensors or not
        :return: None
        """
        if not self.config.USE_RENDERED_OBSERVATIONS:
            super().set_agent_state(position, rotation, agent_id=agent_id, reset_sensors=reset_sensors)
        else:
            pass

    @property
    def current_scene_observation_file(self):
        r"""
        get path to pre-rendered observations for the current scene
        :return: path to pre-rendered observations for the current scene
        """
        return os.path.join(self.config.RENDERED_OBSERVATIONS, self.temp_scene_dataset,
                            self.current_scene_name + '.pkl')

    @property
    def meta_dir(self):
        r"""
        get path to meta-dir containing data about location of navigation nodes and their connectivity
        :return: path to meta-dir containing data about location of navigation nodes and their connectivity
        """
        return os.path.join(self.config.AUDIO.META_DIR, self.current_scene_name)

    @property
    def current_scene_name(self):
        r"""
        get current scene name
        :return: current scene name
        """
        if self.temp_scene_dataset == "mp3d":
            return self._current_scene.split('/')[-2]
        elif self.temp_scene_dataset == "replica":
            return self._current_scene.split('/')[-3]
        else:
            raise ValueError

    def reconfigure(self, config: Config) -> None:
        r"""
        reconfigure for new episode
        :param config: config for reconfiguration
        :return: None
        """
        self.config = config
        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {}'.format(self.current_scene_name))

            if not self.config.USE_RENDERED_OBSERVATIONS:
                self._sim.close()
                del self._sim
                self.sim_config = self.create_sim_config(self._sensor_suite)
                self._sim = habitat_sim.Simulator(self.sim_config)
                self._update_agents_state()
            else:
                with open(self.current_scene_observation_file, 'rb') as fo:
                    self._frame_cache = pickle.load(fo)
            logging.info('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_points_data(self.meta_dir, self.config.AUDIO.GRAPH_FILE,
                                                       scene_dataset=self.temp_scene_dataset)
            for node in self.graph.nodes():
                self._position_to_index_mapping[self.position_encoding(self.graph.nodes()[node]['point'])] = node

        # set agent positions
        self._receiver_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)

        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        self._rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(
                             self.config.AGENT_0.START_ROTATION))[0]))) % 360
        if not self.config.USE_RENDERED_OBSERVATIONS:
            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 self.config.AGENT_0.START_ROTATION)
        else:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_coeffs(self.config.AGENT_0.START_ROTATION))

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            raise ValueError("Position misalignment.")

    def _get_sim_observation(self):
        r"""
        get current observation from simulator
        :return: current observation
        """
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._frame_cache:
            return self._frame_cache[joint_index]
        else:
            sim_obs = self._sim.get_sensor_observations()
            self._frame_cache[joint_index] = sim_obs
            return sim_obs

    def reset(self):
        r"""
        reset simulator for new episode
        :return: None
        """
        logging.debug('Reset simulation')

        if not self.config.USE_RENDERED_OBSERVATIONS:
            sim_obs = self._sim.reset()
            if self._update_agents_state():
                sim_obs = self._get_sim_observation()
        else:
            sim_obs = self._get_sim_observation()
            self._sim.set_sensor_observations(sim_obs)

        self._is_episode_active = True
        self._previous_step_collided = False
        self._prev_sim_obs = sim_obs
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action, only_allowed=True):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :param only_allowed: if true, then can't step anywhere except allowed locations
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        self._previous_step_collided = False

        # PAUSE: 0, FORWARD: 1, LEFT: 2, RIGHT: 3
        if action == HabitatSimActions.MOVE_FORWARD:
            self._previous_step_collided = True
            # the agent initially faces -Z by default
            for neighbor in self.graph[self._receiver_position_index]:
                p1 = self.graph.nodes[self._receiver_position_index]['point']
                p2 = self.graph.nodes[neighbor]['point']
                direction = int(np.around(np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                if direction not in [0, 90, 180, 270]:
                    # diagonal connection
                    if int(abs(direction - self.get_orientation())) == 45:
                        self._receiver_position_index = neighbor
                        self._previous_step_collided = False
                        break
                elif direction == self.get_orientation():
                    self._receiver_position_index = neighbor
                    self._previous_step_collided = False
                    break
        elif action == HabitatSimActions.TURN_LEFT:
            # agent rotates counterclockwise, so turning left means increasing rotation angle by 90
            self._rotation_angle = (self._rotation_angle + 90) % 360
        elif action == HabitatSimActions.TURN_RIGHT:
            self._rotation_angle = (self._rotation_angle - 90) % 360
        elif action == HabitatSimActions.PAUSE:
            raise ValueError
            pass
        else:
            raise NotImplementedError(str(action) + " not in action space -- [PAUSE: 0, MOVE_FORWARD: 1, TURN_LEFT: 2,"
                                                    "TURN_RIGHT: 3]")

        if not self.config.USE_RENDERED_OBSERVATIONS:
            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
        else:
            self._sim.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                      quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))

        # log debugging info
        logging.debug('After taking action {}, r: {}, orientation: {}, location: {}'.format(
            action, self._receiver_position_index, self.get_orientation(),
            self.graph.nodes[self._receiver_position_index]['point']))

        sim_obs = self._get_sim_observation()
        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_sensor_observations(sim_obs)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def get_orientation(self):
        r"""
        get current orientation of the agent
        :return: current orientation of the agent
        """
        _base_orientation = 270
        return (_base_orientation - self._rotation_angle) % 360

    def write_info_to_obs(self, observations):
        r"""
        write agent location and orientation info, and scene name to observation dict... probably
        redundant
        :param observations: observation dict containing different info about the current observation
        :return: None
        """
        observations["agent node and location"] = (self._receiver_position_index,
                                                   self.graph.nodes[self._receiver_position_index]["point"])
        observations["scene name"] = self.current_scene_name
        observations["orientation"] = self._rotation_angle

    @property
    def azimuth_angle(self):
        r"""
        get current azimuth of the agent
        :return: current azimuth of the agent
        """
        # this is the angle used to index the binaural audio files
        # in mesh coordinate systems, +Y forward, +X rightward, +Z upward
        # azimuth is calculated clockwise so +Y is 0 and +X is 90
        return -(self._rotation_angle + 0) % 360

    def geodesic_distance(self, position_a, position_b):
        r"""
        get geodesic distance between 2 nodes
        :param position_a: position of 1st node
        :param position_b: position of 2nd node
        :return: geodesic distance between 2 nodes
        """
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_b)
        assert index_a is not None and index_b is not None
        steps = nx.shortest_path_length(self.graph, index_a, index_b) * self.config.GRID_SIZE
        return steps

    def euclidean_distance(self, position_a, position_b):
        r"""
        get euclidean distance between 2 nodes
        :param position_a: position of 1st node
        :param position_b: position of 2nd node
        :return: euclidean distance between 2 nodes
        """
        assert len(position_a) == len(position_b) == 3
        assert position_a[1] == position_b[1], "height should be same for node a and b"
        return np.power(np.power(position_a[0] - position_b[0],  2) + np.power(position_a[2] - position_b[2], 2), 0.5)

    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    def get_current_bin_spec_mag(self):
        raise NotImplementedError

    def get_current_spatial_mono_spec_mag(self):
        raise NotImplementedError
