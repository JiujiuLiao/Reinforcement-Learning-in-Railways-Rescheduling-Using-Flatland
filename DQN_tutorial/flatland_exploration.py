import numpy as np
import time
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.utils.rendertools import RenderTool

class ObservePredictions(TreeObsForRailEnv):
    """
    We use the provided ShortestPathPredictor to illustrate the usage of predictors in your custom observation.

    We derive our observation builder from TreeObsForRailEnv, to exploit the existing implementation to compute
    the minimum distances from each grid node to each agent's target.

    This is necessary so that we can pass the distance map to the ShortestPathPredictor

    Here we also want to highlight how you can visualize your observation
    """

    def __init__(self, predictor):
        super().__init__(max_depth=0)
        self.predictor = predictor

    def reset(self):
        # Recompute the distance map, if the environment has changed.
        super().reset()

    def get_many(self, handles=None):
        '''
        Because we do not want to call the predictor seperately for every agent we implement the get_many function
        Here we can call the predictor just once for all the agents and use the predictions to generate our observations
        :param handles:
        :return:
        '''

        self.predictions = self.predictor.get()

        self.predicted_pos = {}
        for t in range(len(self.predictions[0])):
            pos_list = []
            for a in handles:
                pos_list.append(self.predictions[a][t][1:3])
            # We transform (x,y) coordinates to a single integer number for simpler comparison
            self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
        observations = {}

        # Collect all the different observation for all the agents
        for h in handles:
            observations[h] = self.get(h)
        return observations

    def get(self, handle):
        '''
        Lets write a simple observation which just indicates whether or not the own predicted path
        overlaps with other predicted paths at any time. This is useless for the task of navigation but might
        help when looking for conflicts. A more complex implementation can be found in the TreeObsForRailEnv class

        Each agent receives an observation of length 10, where each element represents a prediction step and its value
        is:
         - 0 if no overlap is happening
         - 1 if any other paths is crossing the predicted cell

        :param handle: handled as an index of an agent
        :return: Observation of handle
        '''

        observation = np.zeros(10)

        # We track what cells where considered while building the observation and make them accessible for rendering
        visited = set()

        for _idx in range(10):
            # Check if any of the other prediction overlap with agents own predictions
            x_coord = self.predictions[handle][_idx][1]
            y_coord = self.predictions[handle][_idx][2]

            # We add every observed cell to the observation rendering
            visited.add((x_coord, y_coord))
            if self.predicted_pos[_idx][handle] in np.delete(self.predicted_pos[_idx], handle, 0):
                # We detect if another agent is predicting to pass through the same cell at the same predicted time
                observation[handle] = 1

        # This variable will be access by the renderer to visualize the observation
        self.env.dev_obs_dict[handle] = visited

        return observation
    
    # Create the Predictor
CustomPredictor = ShortestPathPredictorForRailEnv(10)


# Pass the Predictor to the observation builder
CustomObsBuilder = ObservePredictions(CustomPredictor)

# Initiate Environment
env = RailEnv(width=30,
              height=30,
              number_of_agents=1,
              rail_generator=sparse_rail_generator(),
              line_generator=sparse_line_generator(),
              obs_builder_object=CustomObsBuilder)

env.reset()

obs, info = env.reset()
env_renderer = RenderTool(env)

# We render the initial step and show the obsereved cells as colored boxes
env_renderer.render_env(show=True, frames=True, show_observations=True, show_predictions=False)

action_dict = {}
for step in range(100):
    for a in range(env.get_num_agents()):
        action = np.random.randint(0, 5)
        action_dict[a] = action
    obs, all_rewards, done, _ = env.step(action_dict)
    print("Rewards: ", all_rewards, "  [done=", done, "]")
    env_renderer.render_env(show=True, frames=True, show_observations=True, show_predictions=False)
    time.sleep(0.5)