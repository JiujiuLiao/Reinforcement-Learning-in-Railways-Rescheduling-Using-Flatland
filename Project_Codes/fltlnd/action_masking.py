"""
Improved Action Masking for Multi-Agent Coordination in Flatland

Key improvement over v1: Only mask forward movement when the agent has
an alternative (can take a different route or wait productively).
Don't mask if it would cause the agent to block the track forever.

Place this file in your fltlnd/ folder as action_masking.py (replace the old one)
"""

import numpy as np
from flatland.envs.agent_utils import RailAgentStatus


class ActionMasker:
    """
    Improved action masker that:
    1. Only masks when alternatives exist
    2. Considers track topology before forcing yields
    3. Avoids creating livelock situations
    """
    
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def __init__(self, conflict_radius=2, enable_priority_yielding=True, 
                 enable_collision_avoidance=True, conservative_mode=False):
        """
        Args:
            conflict_radius: Distance to check for conflicts (reduced from 3 to 2)
            enable_priority_yielding: If True, lower priority yields to higher
            enable_collision_avoidance: If True, prevent moving into occupied cells
            conservative_mode: If True, be more aggressive with masking (not recommended)
        """
        self.conflict_radius = conflict_radius
        self.enable_priority_yielding = enable_priority_yielding
        self.enable_collision_avoidance = enable_collision_avoidance
        self.conservative_mode = conservative_mode
    
    def get_action_mask(self, env, agent_handle):
        """Get valid action mask for a specific agent."""
        mask = np.ones(5, dtype=np.float32)
        agent = env.agents[agent_handle]
        
        # Handle non-active agents
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            # Can enter (forward) or wait (do nothing)
            # Check if entry cell is blocked
            if self._is_spawn_blocked(env, agent):
                mask = np.array([1, 0, 0, 0, 1], dtype=np.float32)  # Wait or stop
            else:
                mask = np.array([1, 0, 1, 0, 0], dtype=np.float32)  # Can enter
            return mask
        
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
        
        # Priority-based yielding - but ONLY if agent has alternatives
        if self.enable_priority_yielding:
            priority_mask = self._get_smart_priority_mask(
                env, agent_handle, agent, valid_transitions, n_movement_options
            )
            mask = mask * priority_mask
        
        # Ensure at least one action is valid
        if mask.sum() == 0:
            # If truly stuck, allow stopping
            mask[4] = 1.0
            # But also allow forward if it was only masked due to priority
            # (better to cause conflict than permanent deadlock)
            if valid_transitions['forward']:
                mask[2] = 0.5  # Lower weight but still allowed
        
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
    
    def _get_smart_priority_mask(self, env, agent_handle, agent, valid_transitions, n_movement_options):
        """
        Smart priority masking that only forces yields when:
        1. There's a genuine conflict (agents approaching each other)
        2. The yielding agent has somewhere else to go OR
        3. The higher-priority agent can actually pass if we yield
        """
        mask = np.ones(5, dtype=np.float32)
        
        if agent.position is None or agent.direction is None:
            return mask
        
        # Find higher-priority agents nearby
        for other in env.agents:
            if other.handle >= agent_handle:
                continue
            if other.position is None or other.status != RailAgentStatus.ACTIVE:
                continue
            
            dist = (abs(agent.position[0] - other.position[0]) + 
                   abs(agent.position[1] - other.position[1]))
            
            if dist > self.conflict_radius:
                continue
            
            # Check if this is actually a conflict situation
            conflict_type = self._classify_conflict(agent, other, env)
            
            if conflict_type == 'none':
                continue
            
            if conflict_type == 'head_on':
                # Head-on: only yield if we have alternative OR are at a switch
                if n_movement_options > 1:
                    # We can take another route
                    mask[2] = 0
                elif self._is_at_switch(env, agent):
                    # We're at a switch, can potentially reverse or wait for path
                    mask[2] = 0
                # else: don't mask - forcing stop would create permanent deadlock
                
            elif conflict_type == 'same_target':
                # Both going to same cell - lower priority should wait
                if n_movement_options > 1:
                    mask[2] = 0
                # else: let them race (collision avoidance will handle it)
                
            elif conflict_type == 'blocking_path':
                # We might block higher priority's path
                # Only yield if it actually helps
                if self._yielding_helps(env, agent, other):
                    if n_movement_options > 1 or self._can_wait_productively(env, agent, other):
                        mask[2] = 0
        
        return mask
    
    def _classify_conflict(self, agent, other, env):
        """Classify the type of conflict between two agents."""
        if agent.direction is None or other.direction is None:
            return 'none'
        
        # Head-on: facing opposite directions on same track
        if (agent.direction + 2) % 4 == other.direction:
            # Check if they're on the same line
            if self._on_same_track(agent, other):
                return 'head_on'
        
        # Same target: both moving to the same cell
        agent_next = self._get_next_position(agent.position, agent.direction)
        other_next = self._get_next_position(other.position, other.direction)
        if agent_next == other_next:
            return 'same_target'
        
        # Check if agent would block other's path
        if agent_next == other.position:
            return 'blocking_path'
        
        return 'none'
    
    def _on_same_track(self, agent, other):
        """Check if two agents are on the same track segment."""
        # Simplified check: are they aligned horizontally or vertically?
        if agent.direction in [0, 2]:  # N-S
            return agent.position[1] == other.position[1]
        else:  # E-W
            return agent.position[0] == other.position[0]
    
    def _is_at_switch(self, env, agent):
        """Check if agent is at a switch (junction with multiple options)."""
        if agent.position is None or agent.direction is None:
            return False
        
        try:
            transitions = env.rail.get_transitions(
                agent.position[0], agent.position[1], agent.direction
            )
            return sum(transitions) > 1
        except:
            return False
    
    def _yielding_helps(self, env, agent, other):
        """Check if this agent yielding would actually help the other agent."""
        # If other agent's next move is blocked by this agent, yielding helps
        if other.direction is None:
            return False
        
        other_next = self._get_next_position(other.position, other.direction)
        return other_next == agent.position
    
    def _can_wait_productively(self, env, agent, other):
        """Check if waiting would eventually resolve (other will pass by)."""
        # Heuristic: if other agent is moving toward their target and we're not
        # directly in their path, waiting might help
        if other.target is None or agent.position is None:
            return False
        
        # Is other agent making progress?
        other_dist_to_target = (abs(other.position[0] - other.target[0]) + 
                                abs(other.position[1] - other.target[1]))
        
        # Simple heuristic: if they're close to target, let them finish
        if other_dist_to_target < 5:
            return True
        
        return False
    
    def get_all_action_masks(self, env):
        """Get action masks for all agents."""
        return {agent.handle: self.get_action_mask(env, agent.handle) 
                for agent in env.agents}


# Also provide a minimal masking option that ONLY does collision avoidance
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
            # Check spawn blocked
            for other in env.agents:
                if other.handle != agent_handle and other.position == agent.initial_position:
                    return np.array([1, 0, 0, 0, 1], dtype=np.float32)
            return np.array([1, 0, 1, 0, 0], dtype=np.float32)
        
        if agent.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
            return np.array([1, 0, 0, 0, 0], dtype=np.float32)
        
        if agent.position is None or agent.direction is None:
            return mask
        
        # Get rail topology
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
        
        # Only collision avoidance - prevent moving into occupied cell
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