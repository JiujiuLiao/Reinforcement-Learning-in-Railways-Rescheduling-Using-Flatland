import numpy as np
from flatland.envs.agent_utils import RailAgentStatus


class ActionMasker:
    """
    Improved action masker that:
    1. Traces along rails to detect true conflicts (not just row/column alignment)
    2. Only masks when it makes sense (considers track topology)
    3. Uses sidings/switches for conflict resolution
    4. Avoids creating permanent deadlocks
    """
    
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def __init__(self, conflict_radius=5, enable_priority_yielding=True, 
                 enable_collision_avoidance=True, conservative_mode=False):
        """
        Args:
            conflict_radius: Distance along track to check for conflicts (increased to 5)
            enable_priority_yielding: If True, lower priority yields to higher
            enable_collision_avoidance: If True, prevent moving into occupied cells
            conservative_mode: If True, be more aggressive with masking
        """
        self.conflict_radius = conflict_radius
        self.enable_priority_yielding = enable_priority_yielding
        self.enable_collision_avoidance = enable_collision_avoidance
        self.conservative_mode = conservative_mode
        
        # Cache for rail tracing results (reset each step)
        self._trace_cache = {}
    
    def reset_cache(self):
        """Call this at the start of each environment step."""
        self._trace_cache = {}
    
    def get_action_mask(self, env, agent_handle):
        """Get valid action mask for a specific agent."""
        mask = np.ones(5, dtype=np.float32)
        agent = env.agents[agent_handle]
        
        # Handle non-active agents
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            if self._is_spawn_blocked(env, agent):
                return np.array([1, 0, 0, 0, 1], dtype=np.float32)  # Wait or stop
            else:
                return np.array([1, 0, 1, 0, 0], dtype=np.float32)  # Can enter
        
        if agent.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
            return np.array([1, 0, 0, 0, 0], dtype=np.float32)
        
        if agent.position is None:
            return mask
        
        # Get valid transitions from rail topology
        valid_transitions = self._get_valid_transitions(env, agent)
        
        # Count how many movement options the agent has
        n_movement_options = sum([valid_transitions['left'], 
                                   valid_transitions['forward'], 
                                   valid_transitions['right']])
        
        # Mask based on rail topology
        if not valid_transitions['left']:
            mask[1] = 0
        if not valid_transitions['forward']:
            mask[2] = 0
        if not valid_transitions['right']:
            mask[3] = 0
        
        # Collision avoidance - don't move into occupied cells
        if self.enable_collision_avoidance:
            collision_mask = self._get_collision_mask(env, agent)
            mask = mask * collision_mask
        
        # Priority-based yielding with improved logic
        if self.enable_priority_yielding:
            priority_mask = self._get_improved_priority_mask(
                env, agent_handle, agent, valid_transitions, n_movement_options
            )
            mask = mask * priority_mask
        
        # Ensure at least one action is valid
        if mask.sum() == 0:
            # Truly stuck - only allow stopping
            mask[4] = 1.0
            # DO NOT allow forward with reduced weight - this defeats the purpose
            # If we're in a deadlock, the deadlock detector will handle it
        
        return mask
    
    def _is_spawn_blocked(self, env, agent):
        """Check if agent's spawn position is blocked."""
        spawn_pos = agent.initial_position
        for other in env.agents:
            if other.handle != agent.handle and other.position == spawn_pos:
                return True
        return False
    
    def _get_valid_transitions(self, env, agent):
        """Get valid transitions based on rail topology."""
        pos = agent.position
        direction = agent.direction
        result = {'left': False, 'forward': False, 'right': False}
        
        if pos is None or direction is None:
            return result
        
        try:
            transitions = env.rail.get_transitions(pos[0], pos[1], direction)
        except:
            return result
        
        left_dir = (direction - 1) % 4
        forward_dir = direction
        right_dir = (direction + 1) % 4
        
        result['left'] = transitions[left_dir] == 1
        result['forward'] = transitions[forward_dir] == 1
        result['right'] = transitions[right_dir] == 1
        
        return result
    
    def _get_next_position(self, position, direction):
        """Get next cell position given current position and direction."""
        dr, dc = self.DIRECTIONS[direction]
        return (position[0] + dr, position[1] + dc)
    
    def _get_collision_mask(self, env, agent):
        """Mask actions that would lead to immediate collision."""
        mask = np.ones(5, dtype=np.float32)
        
        if agent.position is None or agent.direction is None:
            return mask
        
        direction = agent.direction
        action_directions = {
            1: (direction - 1) % 4,
            2: direction,
            3: (direction + 1) % 4,
        }
        
        for action, move_dir in action_directions.items():
            next_pos = self._get_next_position(agent.position, move_dir)
            for other in env.agents:
                if other.handle != agent.handle and other.position == next_pos:
                    mask[action] = 0
                    break
        
        return mask
    
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
            if not self._is_valid_position(env, next_pos):
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
            # Get transitions at new position from opposite direction (we're entering)
            try:
                entering_dir = (current_dir + 2) % 4  # Direction we're coming from
                # Find where we can go from here
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
        For straight track: continue in same direction
        For curves: follow the curve
        For switches: prefer straight, then check alternatives
        """
        try:
            # Check all possible outgoing directions
            possible_dirs = []
            for out_dir in range(4):
                if out_dir == (incoming_dir + 2) % 4:
                    continue  # Skip reverse direction
                transitions = env.rail.get_transitions(position[0], position[1], incoming_dir)
                if transitions[out_dir] == 1:
                    possible_dirs.append(out_dir)
            
            if not possible_dirs:
                return None
            
            # Prefer continuing straight if possible
            if incoming_dir in possible_dirs:
                return incoming_dir
            
            # Otherwise take first available direction
            return possible_dirs[0]
        except:
            return None
    
    def _is_valid_position(self, env, position):
        """Check if position is within map bounds."""
        try:
            return (0 <= position[0] < env.rail.height and 
                    0 <= position[1] < env.rail.width)
        except:
            return False
    
    def _find_agents_on_track(self, env, agent, path):
        """
        Find other agents along the traced track path.
        Returns list of (other_agent, distance, is_head_on) tuples.
        """
        found_agents = []
        agent_handle = agent.handle
        
        for dist, (pos, direction) in enumerate(path, 1):
            for other in env.agents:
                if other.handle == agent_handle:
                    continue
                if other.position != pos:
                    continue
                if other.status != RailAgentStatus.ACTIVE:
                    continue
                
                # Check if head-on (facing opposite direction)
                is_head_on = False
                if other.direction is not None:
                    is_head_on = (other.direction == (direction + 2) % 4)
                
                found_agents.append((other, dist, is_head_on))
        
        return found_agents
    
    def _agent_can_yield(self, env, agent):
        """
        Check if agent is at a position where it can yield (take alternate route).
        This includes switches and sidings.
        """
        if agent.position is None or agent.direction is None:
            return False
        
        try:
            transitions = env.rail.get_transitions(
                agent.position[0], agent.position[1], agent.direction
            )
            n_options = sum(transitions)
            return n_options > 1
        except:
            return False
    
    def _get_improved_priority_mask(self, env, agent_handle, agent, valid_transitions, n_movement_options):
        """
        Improved priority masking that:
        1. Traces along the rail to find true conflicts
        2. Works even on straight track by detecting upcoming conflicts
        3. Uses siding detection for smart yielding decisions
        4. Handles mutual deadlocks properly
        """
        mask = np.ones(5, dtype=np.float32)
        
        if agent.position is None or agent.direction is None:
            return mask
        
        # Trace forward along the track
        forward_path = self._trace_track_forward(
            env, agent.position, agent.direction, self.conflict_radius
        )
        
        # Find agents along our path
        agents_ahead = self._find_agents_on_track(env, agent, forward_path)
        
        # Check if this agent can yield (is at a switch/siding)
        i_can_yield = self._agent_can_yield(env, agent) or n_movement_options > 1
        
        for other, dist, is_head_on in agents_ahead:
            # Skip lower-priority agents (they should yield to us)
            if other.handle > agent_handle:
                continue
            
            # Higher-priority agent found ahead
            other_can_yield = self._agent_can_yield(env, other)
            
            if is_head_on:
                # HEAD-ON CONFLICT
                # Determine who should yield based on:
                # 1. Priority (lower handle = higher priority)
                # 2. Ability to yield (who is at a switch)
                # 3. Distance to target (who is closer to finishing)
                
                should_i_yield = self._should_yield_in_conflict(
                    agent, other, i_can_yield, other_can_yield
                )
                
                if should_i_yield:
                    if i_can_yield:
                        # I can take alternate route - mask forward
                        mask[2] = 0
                    elif dist <= 2:
                        # Very close and I can't yield - must stop
                        # This prevents collision but may cause deadlock
                        # which is better than crash
                        mask[2] = 0
                    # else: I can't yield and they're not too close - keep moving
                    # hoping they find a siding first
                    
            else:
                # SAME DIRECTION or CONVERGING
                # If higher priority agent is ahead going same way, we might
                # need to slow down to avoid rear-ending them
                if dist <= 2:
                    # Too close - consider stopping to avoid collision
                    if n_movement_options > 1:
                        # We have alternatives
                        mask[2] = 0
        
        # Check for agents entering from sides (same target cell)
        self._check_lateral_conflicts(env, agent_handle, agent, mask, valid_transitions)
        
        return mask
    
    def _should_yield_in_conflict(self, agent, other, i_can_yield, other_can_yield):
        """
        Determine if this agent should yield in a conflict with 'other'.
        
        Priority rules:
        1. Agent handle (lower = higher priority) - other has priority over agent
        2. If one can yield and other can't, yielder yields
        3. If both can/can't yield, lower priority yields
        """
        # other.handle < agent.handle, so other has priority
        
        # Case 1: Only I can yield - I should yield
        if i_can_yield and not other_can_yield:
            return True
        
        # Case 2: Only they can yield - they should yield (I don't)
        if not i_can_yield and other_can_yield:
            return False
        
        # Case 3: Both or neither can yield - lower priority (higher handle) yields
        # Since other.handle < agent.handle, I am lower priority, so I yield
        return True
    
    def _check_lateral_conflicts(self, env, agent_handle, agent, mask, valid_transitions):
        """
        Check for conflicts where two agents would move into the same cell.
        """
        if agent.position is None or agent.direction is None:
            return
        
        direction = agent.direction
        my_next_positions = {}
        
        # Calculate where each of my actions would take me
        if valid_transitions['left']:
            left_dir = (direction - 1) % 4
            my_next_positions[1] = self._get_next_position(agent.position, left_dir)
        if valid_transitions['forward']:
            my_next_positions[2] = self._get_next_position(agent.position, direction)
        if valid_transitions['right']:
            right_dir = (direction + 1) % 4
            my_next_positions[3] = self._get_next_position(agent.position, right_dir)
        
        # Check if any higher-priority agent would move into the same cell
        for other in env.agents:
            if other.handle >= agent_handle:
                continue
            if other.position is None or other.direction is None:
                continue
            if other.status != RailAgentStatus.ACTIVE:
                continue
            
            # Where would the other agent go?
            other_next = self._get_next_position(other.position, other.direction)
            
            # Check if any of my moves would conflict
            for action, my_next in my_next_positions.items():
                if my_next == other_next:
                    # Conflict! Higher priority gets the cell, I should wait
                    mask[action] = 0
    
    def get_all_action_masks(self, env):
        """Get action masks for all agents."""
        # Reset trace cache at start of each step
        self.reset_cache()
        return {agent.handle: self.get_action_mask(env, agent.handle) 
                for agent in env.agents}


class MinimalActionMasker:
    """
    Minimal masker that only prevents immediate collisions.
    No priority-based yielding - let the RL agent learn coordination.
    """
    
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def __init__(self):
        pass
    
    def get_action_mask(self, env, agent_handle):
        """Only mask actions that cause immediate collisions."""
        mask = np.ones(5, dtype=np.float32)
        agent = env.agents[agent_handle]
        
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            for other in env.agents:
                if other.handle != agent_handle and other.position == agent.initial_position:
                    return np.array([1, 0, 0, 0, 1], dtype=np.float32)
            return np.array([1, 0, 1, 0, 0], dtype=np.float32)
        
        if agent.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
            return np.array([1, 0, 0, 0, 0], dtype=np.float32)
        
        if agent.position is None or agent.direction is None:
            return mask
        
        try:
            transitions = env.rail.get_transitions(
                agent.position[0], agent.position[1], agent.direction
            )
        except:
            return mask
        
        direction = agent.direction
        left_dir = (direction - 1) % 4
        right_dir = (direction + 1) % 4
        
        if transitions[left_dir] != 1:
            mask[1] = 0
        if transitions[direction] != 1:
            mask[2] = 0
        if transitions[right_dir] != 1:
            mask[3] = 0
        
        action_directions = {
            1: left_dir,
            2: direction,
            3: right_dir,
        }
        
        for action, move_dir in action_directions.items():
            if mask[action] == 0:
                continue
            dr, dc = self.DIRECTIONS[move_dir]
            next_pos = (agent.position[0] + dr, agent.position[1] + dc)
            for other in env.agents:
                if other.handle != agent_handle and other.position == next_pos:
                    mask[action] = 0
                    break
        
        if mask.sum() == 0:
            mask[4] = 1.0
        
        return mask
    
    def get_all_action_masks(self, env):
        return {agent.handle: self.get_action_mask(env, agent.handle) 
                for agent in env.agents}