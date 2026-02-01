"""
Staggered Departures Implementation for Flatland
================================================

This module provides utilities for implementing staggered train departures,
which is the primary recommended solution for the multi-agent deadlock problem.

The core insight: When all agents spawn at t=0, they immediately compete for
limited track resources, causing unavoidable deadlocks. By staggering departures,
we give each agent time to move before the next one enters.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class StaggeredDepartureController:
    """
    Controls when agents are allowed to depart based on a staggered schedule.
    
    This controller works by:
    1. Computing departure times for each agent based on spacing
    2. Masking out all actions except DO_NOTHING for agents before their departure time
    3. Optionally providing rewards/penalties related to departure timing
    """
    
    def __init__(
        self,
        n_agents: int,
        departure_spacing: int = 10,
        spacing_mode: str = 'fixed',
        random_window: int = 5
    ):
        """
        Initialize the staggered departure controller.
        
        Args:
            n_agents: Number of agents in the environment
            departure_spacing: Base time steps between departures
            spacing_mode: 'fixed', 'random', or 'adaptive'
                - 'fixed': Agent i departs at t = i * departure_spacing
                - 'random': Agent i departs at t = i * departure_spacing + random(0, random_window)
                - 'adaptive': Spacing based on map size and agent count
            random_window: For 'random' mode, the window size for randomization
        """
        self.n_agents = n_agents
        self.departure_spacing = departure_spacing
        self.spacing_mode = spacing_mode
        self.random_window = random_window
        
        # Compute departure times
        self.departure_times = self._compute_departure_times()
        
    def _compute_departure_times(self) -> Dict[int, int]:
        """Compute departure time for each agent."""
        departure_times = {}
        
        if self.spacing_mode == 'fixed':
            for i in range(self.n_agents):
                departure_times[i] = i * self.departure_spacing
                
        elif self.spacing_mode == 'random':
            for i in range(self.n_agents):
                base_time = i * self.departure_spacing
                random_offset = np.random.randint(0, self.random_window + 1)
                departure_times[i] = base_time + random_offset
                
        elif self.spacing_mode == 'adaptive':
            # More agents = more spacing needed
            adaptive_spacing = max(5, self.departure_spacing * (self.n_agents / 3))
            for i in range(self.n_agents):
                departure_times[i] = int(i * adaptive_spacing)
        
        return departure_times
    
    def reset(self):
        """Reset departure times (call at start of each episode)."""
        self.departure_times = self._compute_departure_times()
        
    def can_depart(self, agent_id: int, current_step: int) -> bool:
        """Check if an agent is allowed to depart at the current step."""
        return current_step >= self.departure_times.get(agent_id, 0)
    
    def get_departure_mask(self, current_step: int) -> Dict[int, bool]:
        """Get a mask indicating which agents can depart."""
        return {
            agent_id: self.can_depart(agent_id, current_step)
            for agent_id in range(self.n_agents)
        }
    
    def mask_actions_for_departure(
        self,
        action_dict: Dict[int, int],
        current_step: int,
        agent_states: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Modify actions to enforce staggered departures.
        
        Agents that haven't reached their departure time are forced to DO_NOTHING (0).
        This only applies to agents in READY_TO_DEPART state.
        
        Args:
            action_dict: Original action dictionary
            current_step: Current environment step
            agent_states: Dictionary mapping agent_id to RailAgentStatus
            
        Returns:
            Modified action dictionary with pre-departure agents set to DO_NOTHING
        """
        # RailAgentStatus values:
        # 0 = READY_TO_DEPART
        # 1 = ACTIVE  
        # 2 = DONE
        # 3 = DONE_REMOVED
        
        READY_TO_DEPART = 0
        DO_NOTHING = 0
        
        modified_actions = dict(action_dict)
        
        for agent_id, action in action_dict.items():
            # Only modify agents that are waiting to depart
            if agent_states.get(agent_id) == READY_TO_DEPART:
                if not self.can_depart(agent_id, current_step):
                    modified_actions[agent_id] = DO_NOTHING
                    
        return modified_actions


def compute_recommended_spacing(
    n_agents: int,
    map_width: int,
    map_height: int,
    base_spacing: int = 8
) -> int:
    """
    Compute recommended departure spacing based on environment parameters.
    
    The formula considers:
    - More agents need more spacing
    - Larger maps can tolerate tighter spacing
    - Base spacing as minimum
    
    Args:
        n_agents: Number of agents
        map_width: Map width in cells
        map_height: Map height in cells
        base_spacing: Minimum spacing between departures
        
    Returns:
        Recommended departure spacing in time steps
    """
    map_area = map_width * map_height
    density = n_agents / map_area
    
    # Higher density = more spacing needed
    # Typical values: 5 agents on 40x27 = 0.0046 density
    density_factor = 1 + (density * 1000)  # Scale up for reasonable values
    
    # More agents = more spacing
    agent_factor = 1 + (n_agents - 1) * 0.2
    
    recommended = int(base_spacing * density_factor * agent_factor)
    
    # Clamp to reasonable range
    return max(base_spacing, min(recommended, 30))


# Recommended spacing values based on your experiments
RECOMMENDED_SPACINGS = {
    1: 0,    # Single agent - no spacing needed
    2: 8,    # Two agents - minimal spacing
    3: 10,   # Three agents - moderate spacing
    5: 15,   # Five agents - significant spacing
    7: 18,   # Seven agents
    10: 22,  # Ten agents - large spacing
}


def get_spacing_for_phase(n_agents: int) -> int:
    """Get recommended spacing for a given number of agents."""
    if n_agents in RECOMMENDED_SPACINGS:
        return RECOMMENDED_SPACINGS[n_agents]
    
    # Interpolate for unlisted values
    sorted_keys = sorted(RECOMMENDED_SPACINGS.keys())
    for i, k in enumerate(sorted_keys[:-1]):
        if k < n_agents < sorted_keys[i + 1]:
            # Linear interpolation
            ratio = (n_agents - k) / (sorted_keys[i + 1] - k)
            return int(
                RECOMMENDED_SPACINGS[k] + 
                ratio * (RECOMMENDED_SPACINGS[sorted_keys[i + 1]] - RECOMMENDED_SPACINGS[k])
            )
    
    # For very large agent counts
    return min(30, 10 + n_agents * 2)


if __name__ == "__main__":
    # Demo usage
    print("Staggered Departures Configuration Demo")
    print("=" * 50)
    
    for n_agents in [1, 2, 3, 5, 7, 10]:
        spacing = get_spacing_for_phase(n_agents)
        computed = compute_recommended_spacing(n_agents, 40, 27)
        
        print(f"\n{n_agents} agents:")
        print(f"  Recommended spacing: {spacing} steps")
        print(f"  Computed spacing: {computed} steps")
        
        controller = StaggeredDepartureController(n_agents, spacing)
        print(f"  Departure times: {controller.departure_times}")
        
        # Show when each agent can depart
        print(f"  Agent 0 departs at: t={controller.departure_times[0]}")
        if n_agents > 1:
            print(f"  Agent {n_agents-1} departs at: t={controller.departure_times[n_agents-1]}")