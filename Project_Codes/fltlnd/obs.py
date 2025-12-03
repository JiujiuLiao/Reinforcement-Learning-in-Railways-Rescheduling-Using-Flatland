from abc import ABC, abstractmethod

import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from fltlnd.utils import split_tree_into_feature_groups, norm_obs_clip


class Observation(ABC):
    def __init__(self, parameters, predictor=None):
        self.parameters = parameters
        self.create(predictor)

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def get_state_size(self):
        pass

    @abstractmethod
    def normalize(self, observation):
        pass


class TreeObs(Observation):
    def create(self, predictor):
        self.builder = TreeObsForRailEnv(max_depth=self.parameters['tree_depth'], predictor=predictor)

    def get_state_size(self):
        # Calculate the state size given the depth of the tree observation and the number of features
        n_features_per_node = self.builder.observation_dim
        n_nodes = 0
        for i in range(self.parameters['tree_depth'] + 1):
            n_nodes += np.power(4, i)
        return n_features_per_node * n_nodes

    def normalize(self, observation, env_handler, agent_handle):
        data, distance, agent_data = split_tree_into_feature_groups(observation, self.parameters['tree_depth'])

        data = norm_obs_clip(data, fixed_radius=self.parameters['radius'])
        distance = norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        normalized_obs = np.concatenate(
            (np.concatenate((data, distance)), agent_data)
        )
        # ---- ADD GLOBAL FEATURES HERE ----
        normalized_obs = self.add_global_features(
            normalized_obs, env_handler, agent_handle
        )

        return normalized_obs

    def add_global_features(self, obs_vector, env_handler, agent):
        """
        Add global features to observation vector:
        - percentage of agents arrived
        - fraction of steps elapsed
        - agent speed
        """

        num_agents = env_handler.get_num_agents()
        arrived = sum([int(a.status == 4) for a in env_handler.env.agents])
        frac_arrived = arrived / num_agents

        # current step / max steps
        step_fraction = env_handler.env._elapsed_steps / env_handler.env._max_episode_steps

        # agent speed
        speed = env_handler.env.agents[agent].speed_data["speed"]

        globals_vec = np.array([frac_arrived, step_fraction, speed], dtype=float)

        return np.concatenate([obs_vector, globals_vec])

