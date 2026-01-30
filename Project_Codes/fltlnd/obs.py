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
    
    This class adds new features on top of the base tree observation
    to help agents understand their relationship to other agents and
    make better coordination decisions.
    
    Key improvements over basic observations:
    1. Rail-traced conflict detection (not just row/column alignment)
    2. Siding/switch awareness for yielding decisions
    3. Better priority signaling
    4. Proactive head-on collision detection
    """
    
    # Direction vectors: UP, RIGHT, DOWN, LEFT (matching Flatland convention)
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def create(self, predictor):
        self.builder = TreeObsForRailEnv(
            max_depth=self.parameters['tree_depth'], 
            predictor=predictor
        )
        # Number of global features: original 3 + new coordination features
        # Features [0-2]: Original (frac_arrived, step_fraction, speed)
        # Features [3-17]: New coordination features (15 total)
        # Total: 18 global features
        self.n_global_features = 18
        
        # For tracking movement (to detect stuck agents)
        self._last_positions = {}
        self._steps_since_move = {}
        
        # Cache for rail tracing (reset each step)
        self._trace_cache = {}

    def get_state_size(self):
        """Calculate total observation size including enhanced features."""
        n_features_per_node = self.builder.observation_dim
        n_nodes = sum(4**i for i in range(self.parameters['tree_depth'] + 1))
        base_size = n_features_per_node * n_nodes
        # Note: handler.py should add self.n_global_features to this
        return base_size

    def reset(self):
        """Reset tracking variables at episode start."""
        self._last_positions = {}
        self._steps_since_move = {}
        self._trace_cache = {}

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
        
        Features (18 total):
        [0]  frac_arrived         - Fraction of agents that completed
        [1]  step_fraction        - Episode progress (0 to 1)
        [2]  speed                - Agent's speed
        [3]  priority             - Agent's priority (lower handle = higher priority)
        [4]  norm_dist_target     - Normalized distance to target
        [5]  nearby_density       - Fraction of other agents nearby
        [6]  is_blocked           - 1.0 if cell ahead is occupied
        [7]  head_on_risk         - 1.0 if facing an agent coming toward us on track
        [8]  head_on_distance     - Distance to head-on agent (normalized, 1.0 if none)
        [9]  agents_ahead         - Number of agents ahead on path (normalized)
        [10] agents_behind        - Number of agents behind (normalized)
        [11] nearest_dist         - Distance to nearest agent (normalized)
        [12] at_switch            - 1.0 if agent is at a switch/siding
        [13] should_yield         - 1.0 if this agent should yield in conflicts
        [14] opponent_at_switch   - 1.0 if head-on opponent is at a switch
        [15] malfunction          - 1.0 if agent is malfunctioning
        [16] steps_stationary     - Steps since last movement (normalized)
        [17] conflict_urgency     - How urgent is the conflict (0=none, 1=imminent)
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
        # NEW COORDINATION FEATURES (3-17)
        # =====================================================================
        
        # [3] Priority based on agent handle (lower = higher priority)
        priority = agent_handle / max(1, num_agents - 1) if num_agents > 1 else 0.5
        
        # [4] Normalized distance to target
        if agent.position is not None and agent.target is not None:
            dist_to_target = (abs(agent.position[0] - agent.target[0]) + 
                            abs(agent.position[1] - agent.target[1]))
            norm_dist_target = min(1.0, dist_to_target / map_diagonal)
        else:
            norm_dist_target = 1.0
        
        # [5] Nearby agent density
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
        
        # [7] & [8] Head-on collision risk using RAIL TRACING
        head_on_risk = 0.0
        head_on_distance = 1.0  # Normalized, 1.0 means no head-on agent
        head_on_agent = None
        
        if agent.position is not None and agent.direction is not None:
            # Trace along the track to find head-on agents
            forward_path = self._trace_track_forward(env, agent.position, agent.direction, 5)
            
            for dist, (pos, path_dir) in enumerate(forward_path, 1):
                for other in env.agents:
                    if other.handle == agent_handle:
                        continue
                    if other.position != pos:
                        continue
                    if other.status != RailAgentStatus.ACTIVE:
                        continue
                    if other.direction is None:
                        continue
                    
                    # Check if facing opposite direction (head-on)
                    if other.direction == (path_dir + 2) % 4:
                        head_on_risk = 1.0
                        head_on_distance = dist / 5.0  # Normalize by max distance
                        head_on_agent = other
                        break
                
                if head_on_risk > 0:
                    break
        
        # [9] Agents ahead on track (using rail tracing)
        agents_ahead = 0
        if agent.position is not None and agent.direction is not None:
            forward_path = self._trace_track_forward(env, agent.position, agent.direction, 5)
            for dist, (pos, path_dir) in enumerate(forward_path, 1):
                for other in env.agents:
                    if (other.handle != agent_handle and 
                        other.position == pos and
                        other.status == RailAgentStatus.ACTIVE):
                        agents_ahead += 1
        
        agents_ahead = min(1.0, agents_ahead / max(1, num_agents - 1))
        
        # [10] Agents behind (using simple direction check - less critical)
        agents_behind = 0
        if agent.position is not None and agent.direction is not None:
            behind_dir = (agent.direction + 2) % 4
            for other in env.agents:
                if (other.handle != agent_handle and 
                    other.position is not None and
                    other.status == RailAgentStatus.ACTIVE):
                    
                    diff_row = other.position[0] - agent.position[0]
                    diff_col = other.position[1] - agent.position[1]
                    
                    if self._is_in_direction(diff_row, diff_col, behind_dir):
                        agents_behind += 1
        
        agents_behind = min(1.0, agents_behind / max(1, num_agents - 1))
        
        # [11] Distance to nearest agent (normalized)
        nearest_dist = 1.0
        
        if agent.position is not None:
            min_dist = float('inf')
            
            for other in env.agents:
                if other.handle != agent_handle and other.position is not None:
                    dist = (abs(agent.position[0] - other.position[0]) + 
                           abs(agent.position[1] - other.position[1]))
                    if dist < min_dist:
                        min_dist = dist
            
            if min_dist < float('inf'):
                nearest_dist = min(1.0, min_dist / map_diagonal)
        
        # [12] At switch/siding (can this agent yield?)
        at_switch = 0.0
        if agent.position is not None and agent.direction is not None:
            try:
                transitions = env.rail.get_transitions(
                    agent.position[0], agent.position[1], agent.direction
                )
                if sum(transitions) > 1:
                    at_switch = 1.0
            except:
                pass
        
        # [13] Should yield signal
        # This agent should yield if:
        # - Lower priority (higher handle) AND
        # - In a conflict situation AND
        # - Either can yield OR opponent can't
        should_yield = 0.0
        
        if head_on_risk > 0 and head_on_agent is not None:
            # Check opponent's ability to yield
            opponent_at_switch = self._agent_at_switch(env, head_on_agent)
            
            if agent_handle > head_on_agent.handle:
                # I'm lower priority
                if at_switch > 0 or not opponent_at_switch:
                    # I can yield OR they can't - I should yield
                    should_yield = 1.0
        elif is_blocked > 0 and blocking_agent is not None:
            if agent_handle > blocking_agent.handle:
                should_yield = 1.0
        
        # [14] Opponent at switch (can the head-on opponent yield?)
        opponent_at_switch = 0.0
        if head_on_agent is not None:
            opponent_at_switch = 1.0 if self._agent_at_switch(env, head_on_agent) else 0.0
        
        # [15] Malfunction status
        malfunction = 0.0
        if hasattr(agent, 'malfunction_data') and agent.malfunction_data is not None:
            if agent.malfunction_data.get('malfunction', 0) > 0:
                malfunction = 1.0
        
        # [16] Steps since last movement
        steps_stationary = self._update_movement_tracking(agent_handle, agent.position)
        steps_stationary_norm = min(1.0, steps_stationary / 50.0)
        
        # [17] Conflict urgency (how urgent is the need to act)
        # 0 = no conflict, 0.5 = conflict but time to react, 1.0 = imminent collision
        conflict_urgency = 0.0
        if head_on_risk > 0:
            # Closer distance = higher urgency
            conflict_urgency = 1.0 - head_on_distance
        elif is_blocked > 0:
            conflict_urgency = 0.7  # Blocked but not head-on
        elif nearby_density > 0.3:
            conflict_urgency = 0.3  # Crowded area
        
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
            head_on_distance,       # [8]
            agents_ahead,           # [9]
            agents_behind,          # [10]
            nearest_dist,           # [11]
            at_switch,              # [12]
            should_yield,           # [13]
            opponent_at_switch,     # [14]
            malfunction,            # [15]
            steps_stationary_norm,  # [16]
            conflict_urgency,       # [17]
        ], dtype=np.float32)
        
        return np.concatenate([obs_vector, globals_vec])

    def _get_next_position(self, position, direction):
        """Get the next cell position given current position and direction."""
        dr, dc = self.DIRECTIONS[direction]
        return (position[0] + dr, position[1] + dc)
    
    def _is_in_direction(self, diff_row, diff_col, direction):
        """
        Check if a relative position (diff_row, diff_col) is in the given direction.
        Uses a cone-based check.
        """
        if diff_row == 0 and diff_col == 0:
            return False
        
        if direction == 0:  # UP
            return diff_row < 0 and abs(diff_col) <= abs(diff_row)
        elif direction == 1:  # RIGHT
            return diff_col > 0 and abs(diff_row) <= abs(diff_col)
        elif direction == 2:  # DOWN
            return diff_row > 0 and abs(diff_col) <= abs(diff_row)
        elif direction == 3:  # LEFT
            return diff_col < 0 and abs(diff_row) <= abs(diff_col)
        
        return False
    
    def _trace_track_forward(self, env, start_pos, start_dir, max_distance):
        """
        Trace along the rail track from a starting position/direction.
        Returns list of (position, direction) tuples along the track.
        
        This properly follows the rail topology instead of just checking
        row/column alignment.
        """
        cache_key = (start_pos, start_dir, max_distance)
        if cache_key in self._trace_cache:
            return self._trace_cache[cache_key]
        
        path = []
        current_pos = start_pos
        current_dir = start_dir
        
        for _ in range(max_distance):
            # Get next position
            next_pos = self._get_next_position(current_pos, current_dir)
            
            # Check if valid position on map
            try:
                if not (0 <= next_pos[0] < env.rail.height and 
                        0 <= next_pos[1] < env.rail.width):
                    break
            except:
                break
            
            # Check if there's a rail connection
            try:
                transitions = env.rail.get_transitions(current_pos[0], current_pos[1], current_dir)
                if transitions[current_dir] != 1:
                    break
            except:
                break
            
            path.append((next_pos, current_dir))
            
            # Determine next direction (follow the rail)
            try:
                next_dir = self._get_continuation_direction(env, next_pos, current_dir)
                if next_dir is None:
                    break
                current_dir = next_dir
            except:
                break
            
            current_pos = next_pos
        
        self._trace_cache[cache_key] = path
        return path
    
    def _get_continuation_direction(self, env, position, incoming_dir):
        """
        Get the direction an agent would continue in after entering a cell.
        """
        try:
            possible_dirs = []
            for out_dir in range(4):
                if out_dir == (incoming_dir + 2) % 4:
                    continue  # Skip reverse
                transitions = env.rail.get_transitions(position[0], position[1], incoming_dir)
                if transitions[out_dir] == 1:
                    possible_dirs.append(out_dir)
            
            if not possible_dirs:
                return None
            
            # Prefer continuing straight if possible
            if incoming_dir in possible_dirs:
                return incoming_dir
            
            return possible_dirs[0]
        except:
            return None
    
    def _agent_at_switch(self, env, agent):
        """Check if an agent is at a switch/siding position."""
        if agent.position is None or agent.direction is None:
            return False
        
        try:
            transitions = env.rail.get_transitions(
                agent.position[0], agent.position[1], agent.direction
            )
            return sum(transitions) > 1
        except:
            return False
    
    def _update_movement_tracking(self, agent_handle, current_position):
        """Track how many steps since the agent last moved."""
        last_pos = self._last_positions.get(agent_handle)
        
        if current_position is None:
            self._steps_since_move[agent_handle] = 0
        elif last_pos is None or last_pos != current_position:
            self._steps_since_move[agent_handle] = 0
        else:
            self._steps_since_move[agent_handle] = self._steps_since_move.get(agent_handle, 0) + 1
        
        self._last_positions[agent_handle] = current_position
        return self._steps_since_move.get(agent_handle, 0)