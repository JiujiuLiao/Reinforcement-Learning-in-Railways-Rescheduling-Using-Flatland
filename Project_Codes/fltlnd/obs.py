from abc import ABC, abstractmethod

import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.agent_utils import RailAgentStatus  
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

class EnhancedTreeObs(Observation):
    """
    Enhanced TreeObs with multi-agent coordination features.
    
    This class adds 12 new features on top of the base tree observation
    to help agents understand their relationship to other agents and
    make better coordination decisions.
    
    The key insight is that deadlocks happen because agents don't know:
    1. Who should yield (priority)
    2. Where other agents are relative to them
    3. Whether they're about to collide
    
    These features provide that information.
    """
    
    # Direction vectors: UP, RIGHT, DOWN, LEFT (matching Flatland convention)
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def create(self, predictor):
        self.builder = TreeObsForRailEnv(
            max_depth=self.parameters['tree_depth'], 
            predictor=predictor
        )
        # Original 3 + new 12 = 15 global features
        self.n_global_features = 15
        
        # For tracking movement (to detect stuck agents)
        self._last_positions = {}
        self._steps_since_move = {}

    def get_state_size(self):
        """Calculate total observation size including enhanced features."""
        n_features_per_node = self.builder.observation_dim
        n_nodes = sum(4**i for i in range(self.parameters['tree_depth'] + 1))
        base_size = n_features_per_node * n_nodes
        # Note: handler.py adds 3 to state_size, but we're adding 15 now
        # So we return base_size, and handler should add self.n_global_features
        return base_size

    def reset(self):
        """Reset tracking variables at episode start."""
        self._last_positions = {}
        self._steps_since_move = {}

    def normalize(self, observation, env_handler, agent_handle):
        """Normalize observation and add enhanced global features."""
        data, distance, agent_data = split_tree_into_feature_groups(
            observation, self.parameters['tree_depth']
        )

        data = norm_obs_clip(data, fixed_radius=self.parameters['radius'])
        distance = norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        
        normalized_obs = np.concatenate([
            np.concatenate([data, distance]), 
            agent_data
        ])
        
        # Add enhanced global features
        normalized_obs = self.add_global_features(
            normalized_obs, env_handler, agent_handle
        )

        return normalized_obs

    def add_global_features(self, obs_vector, env_handler, agent_handle):
        """
        Add enhanced global features for multi-agent coordination.
        
        Features (15 total):
        [0]  frac_arrived      - Fraction of agents that completed
        [1]  step_fraction     - Episode progress (0 to 1)
        [2]  speed             - Agent's speed
        [3]  priority          - Agent's priority (lower handle = higher priority)
        [4]  norm_dist_target  - Normalized distance to target
        [5]  nearby_density    - Fraction of other agents nearby
        [6]  is_blocked        - 1.0 if cell ahead is occupied
        [7]  head_on_risk      - 1.0 if facing an agent coming toward us
        [8]  agents_ahead      - Number of agents ahead on path (normalized)
        [9]  agents_behind     - Number of agents behind (normalized)
        [10] nearest_dist      - Distance to nearest agent (normalized)
        [11] nearest_rel_dir   - Relative direction to nearest agent (-1 to 1)
        [12] should_yield      - 1.0 if this agent should yield in conflicts
        [13] malfunction       - 1.0 if agent is malfunctioning
        [14] steps_stationary  - Steps since last movement (normalized)
        """
        env = env_handler.env
        agent = env.agents[agent_handle]
        num_agents = env_handler.get_num_agents()
        
        # Map dimensions for normalization
        map_diagonal = env_handler.x_dim + env_handler.y_dim
        
        # =====================================================================
        # ORIGINAL FEATURES (0-2)
        # =====================================================================
        
        # [0] Fraction of agents arrived
        arrived = sum(1 for a in env.agents 
                     if a.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED])
        frac_arrived = arrived / num_agents
        
        # [1] Step fraction (episode progress)
        step_fraction = env._elapsed_steps / env._max_episode_steps
        
        # [2] Agent speed
        speed = agent.speed_data["speed"]
        
        # =====================================================================
        # NEW COORDINATION FEATURES (3-14)
        # =====================================================================
        
        # [3] Priority based on agent handle (lower = higher priority)
        # This gives agents a consistent way to decide who yields
        priority = agent_handle / max(1, num_agents - 1) if num_agents > 1 else 0.5
        
        # [4] Normalized distance to target
        if agent.position is not None and agent.target is not None:
            dist_to_target = (abs(agent.position[0] - agent.target[0]) + 
                            abs(agent.position[1] - agent.target[1]))
            norm_dist_target = min(1.0, dist_to_target / map_diagonal)
        else:
            norm_dist_target = 1.0
        
        # [5] Nearby agent density (agents within threshold distance)
        nearby_threshold = max(5, min(env_handler.x_dim, env_handler.y_dim) // 4)
        nearby_count = 0
        
        if agent.position is not None:
            for other in env.agents:
                if other.handle != agent_handle and other.position is not None:
                    dist = (abs(agent.position[0] - other.position[0]) + 
                           abs(agent.position[1] - other.position[1]))
                    if dist <= nearby_threshold:
                        nearby_count += 1
        
        nearby_density = nearby_count / max(1, num_agents - 1)
        
        # [6] Is blocked (cell immediately ahead is occupied)
        is_blocked = 0.0
        blocking_agent = None
        
        if agent.position is not None and agent.direction is not None:
            next_pos = self._get_next_position(agent.position, agent.direction)
            for other in env.agents:
                if other.handle != agent_handle and other.position == next_pos:
                    is_blocked = 1.0
                    blocking_agent = other
                    break
        
        # [7] Head-on collision risk
        # Check if any agent within 3 cells is facing toward us
        head_on_risk = 0.0
        
        if agent.position is not None and agent.direction is not None:
            opposite_dir = (agent.direction + 2) % 4
            
            for other in env.agents:
                if (other.handle != agent_handle and 
                    other.position is not None and 
                    other.direction is not None):
                    
                    dist = (abs(agent.position[0] - other.position[0]) + 
                           abs(agent.position[1] - other.position[1]))
                    
                    # Check if other agent is roughly ahead and facing us
                    if dist <= 3 and other.direction == opposite_dir:
                        # Verify they're actually on a collision course
                        # (in the forward direction, not to the side)
                        diff_row = other.position[0] - agent.position[0]
                        diff_col = other.position[1] - agent.position[1]
                        
                        # Check if other is in our forward direction
                        if self._is_in_direction(diff_row, diff_col, agent.direction):
                            head_on_risk = 1.0
                            break
        
        # [8] Agents ahead on path (simplified: agents in forward cone)
        # [9] Agents behind (agents in backward cone)
        agents_ahead = 0
        agents_behind = 0
        
        if agent.position is not None and agent.direction is not None:
            for other in env.agents:
                if (other.handle != agent_handle and 
                    other.position is not None and
                    other.status == RailAgentStatus.ACTIVE):
                    
                    diff_row = other.position[0] - agent.position[0]
                    diff_col = other.position[1] - agent.position[1]
                    
                    if self._is_in_direction(diff_row, diff_col, agent.direction):
                        agents_ahead += 1
                    elif self._is_in_direction(diff_row, diff_col, (agent.direction + 2) % 4):
                        agents_behind += 1
        
        agents_ahead = min(1.0, agents_ahead / max(1, num_agents - 1))
        agents_behind = min(1.0, agents_behind / max(1, num_agents - 1))
        
        # [10] Distance to nearest agent (normalized)
        # [11] Relative direction to nearest agent
        nearest_dist = 1.0
        nearest_rel_dir = 0.0
        
        if agent.position is not None:
            min_dist = float('inf')
            nearest_other = None
            
            for other in env.agents:
                if other.handle != agent_handle and other.position is not None:
                    dist = (abs(agent.position[0] - other.position[0]) + 
                           abs(agent.position[1] - other.position[1]))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_other = other
            
            if nearest_other is not None:
                nearest_dist = min(1.0, min_dist / map_diagonal)
                
                # Relative direction: -1 (behind) to +1 (ahead)
                if agent.direction is not None:
                    diff_row = nearest_other.position[0] - agent.position[0]
                    diff_col = nearest_other.position[1] - agent.position[1]
                    
                    if self._is_in_direction(diff_row, diff_col, agent.direction):
                        nearest_rel_dir = 1.0  # Ahead
                    elif self._is_in_direction(diff_row, diff_col, (agent.direction + 2) % 4):
                        nearest_rel_dir = -1.0  # Behind
                    else:
                        nearest_rel_dir = 0.0  # Side
        
        # [12] Should this agent yield?
        # Yield if: lower priority AND in potential conflict
        should_yield = 0.0
        
        if is_blocked > 0 and blocking_agent is not None:
            # In a blocking situation, lower priority agent should yield
            if agent_handle > blocking_agent.handle:
                should_yield = 1.0
        elif head_on_risk > 0:
            # In head-on situation, higher handle (lower priority) yields
            should_yield = priority  # Already normalized 0-1
        
        # [13] Malfunction status
        malfunction = 0.0
        if hasattr(agent, 'malfunction_data') and agent.malfunction_data is not None:
            if agent.malfunction_data.get('malfunction', 0) > 0:
                malfunction = 1.0
        
        # [14] Steps since last movement (to detect stuck agents)
        steps_stationary = self._update_movement_tracking(agent_handle, agent.position)
        steps_stationary_norm = min(1.0, steps_stationary / 50.0)  # Normalize to ~50 steps
        
        # =====================================================================
        # COMBINE ALL FEATURES
        # =====================================================================
        
        globals_vec = np.array([
            frac_arrived,           # [0]
            step_fraction,          # [1]
            speed,                  # [2]
            priority,               # [3]
            norm_dist_target,       # [4]
            nearby_density,         # [5]
            is_blocked,             # [6]
            head_on_risk,           # [7]
            agents_ahead,           # [8]
            agents_behind,          # [9]
            nearest_dist,           # [10]
            nearest_rel_dir,        # [11]
            should_yield,           # [12]
            malfunction,            # [13]
            steps_stationary_norm,  # [14]
        ], dtype=np.float32)
        
        return np.concatenate([obs_vector, globals_vec])

    def _get_next_position(self, position, direction):
        """Get the next cell position given current position and direction."""
        dr, dc = self.DIRECTIONS[direction]
        return (position[0] + dr, position[1] + dc)
    
    def _is_in_direction(self, diff_row, diff_col, direction):
        """
        Check if a relative position (diff_row, diff_col) is in the given direction.
        Uses a cone-based check (45 degrees each side of the direction).
        """
        if diff_row == 0 and diff_col == 0:
            return False
            
        # Direction vectors
        # 0: UP (-1, 0), 1: RIGHT (0, 1), 2: DOWN (1, 0), 3: LEFT (0, -1)
        
        if direction == 0:  # UP
            return diff_row < 0 and abs(diff_col) <= abs(diff_row)
        elif direction == 1:  # RIGHT
            return diff_col > 0 and abs(diff_row) <= abs(diff_col)
        elif direction == 2:  # DOWN
            return diff_row > 0 and abs(diff_col) <= abs(diff_row)
        elif direction == 3:  # LEFT
            return diff_col < 0 and abs(diff_row) <= abs(diff_col)
        
        return False
    
    def _update_movement_tracking(self, agent_handle, current_position):
        """Track how many steps since the agent last moved."""
        last_pos = self._last_positions.get(agent_handle)
        
        if current_position is None:
            # Agent not spawned or removed
            self._steps_since_move[agent_handle] = 0
        elif last_pos is None or last_pos != current_position:
            # Agent moved
            self._steps_since_move[agent_handle] = 0
        else:
            # Agent stayed in same position
            self._steps_since_move[agent_handle] = self._steps_since_move.get(agent_handle, 0) + 1
        
        self._last_positions[agent_handle] = current_position
        return self._steps_since_move.get(agent_handle, 0)