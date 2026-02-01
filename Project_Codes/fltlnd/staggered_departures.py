"""
Staggered Departures Module for Flatland Multi-Agent RL

This module implements staggered departure scheduling to reduce initial
track competition and deadlocks in multi-agent railway scenarios.

Key insight: Simultaneous spawning at t=0 causes immediate conflicts.
Staggering departures gives each agent time to clear initial track segments.
"""

from typing import Dict, Optional, List
import numpy as np


# ============================================================
# INCREASED SPACING VALUES
# These are tuned to give agents enough separation
# ============================================================
RECOMMENDED_SPACINGS = {
    1: 0,    # Single agent - no spacing needed
    2: 10,   # Two agents - 10 steps apart (increased from 8)
    3: 15,   # Three agents - 15 steps apart (increased from 12)
    5: 25,   # Five agents - 25 steps apart (increased from 15)
    7: 30,   # Seven agents
    10: 35,  # Ten agents
}


def get_spacing_for_phase(n_agents: int) -> int:
    """
    Get recommended departure spacing based on number of agents.
    
    Args:
        n_agents: Number of agents in the environment
        
    Returns:
        Recommended spacing in steps between departures
    """
    if n_agents in RECOMMENDED_SPACINGS:
        return RECOMMENDED_SPACINGS[n_agents]
    
    # For other agent counts, interpolate/extrapolate
    if n_agents < 2:
        return 0
    elif n_agents <= 3:
        return 12 + (n_agents - 2) * 3
    elif n_agents <= 5:
        return 15 + (n_agents - 3) * 5
    elif n_agents <= 10:
        return 25 + (n_agents - 5) * 2
    else:
        # For very large numbers, use formula
        return min(40, 25 + (n_agents - 5) * 2)


class StaggeredDepartureController:
    """
    Controls staggered departure timing for multi-agent scenarios.
    
    This controller determines when each agent is allowed to depart
    and masks actions to enforce the departure schedule.
    """
    
    def __init__(
        self,
        n_agents: int,
        departure_spacing: int = 15,
        spacing_mode: str = 'fixed'
    ):
        """
        Initialize the staggered departure controller.
        
        Args:
            n_agents: Total number of agents
            departure_spacing: Number of steps between departures
            spacing_mode: 'fixed', 'random', or 'adaptive'
        """
        self.n_agents = n_agents
        self.departure_spacing = departure_spacing
        self.spacing_mode = spacing_mode
        
        # Compute departure times
        self.departure_times = self._compute_departure_times()
        
        # Track which agents have departed
        self.has_departed = {i: False for i in range(n_agents)}
    
    def _compute_departure_times(self) -> Dict[int, int]:
        """Compute departure time for each agent."""
        departure_times = {}
        
        if self.spacing_mode == 'fixed':
            # Simple linear spacing
            for i in range(self.n_agents):
                departure_times[i] = i * self.departure_spacing
                
        elif self.spacing_mode == 'random':
            # Random spacing within bounds
            base_times = [i * self.departure_spacing for i in range(self.n_agents)]
            jitter = self.departure_spacing // 4
            for i in range(self.n_agents):
                departure_times[i] = max(0, base_times[i] + np.random.randint(-jitter, jitter + 1))
                
        elif self.spacing_mode == 'adaptive':
            # Could be extended to adapt based on environment
            for i in range(self.n_agents):
                departure_times[i] = i * self.departure_spacing
        
        return departure_times
    
    def reset(self):
        """Reset the controller for a new episode."""
        self.has_departed = {i: False for i in range(self.n_agents)}
        
        # Recompute departure times if using random mode
        if self.spacing_mode == 'random':
            self.departure_times = self._compute_departure_times()
    
    def can_depart(self, agent_handle: int, current_step: int) -> bool:
        """
        Check if an agent is allowed to depart at the current step.
        
        Args:
            agent_handle: The agent's handle/ID
            current_step: Current environment step
            
        Returns:
            True if agent can depart, False otherwise
        """
        if agent_handle not in self.departure_times:
            return True  # Unknown agent - allow departure
        
        return current_step >= self.departure_times[agent_handle]
    
    def mask_actions_for_departure(
        self,
        action_dict: Dict[int, int],
        current_step: int,
        agent_states: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Modify action dictionary to enforce departure schedule.
        
        Agents that haven't reached their departure time are forced
        to take DO_NOTHING (action 0) instead of entering the environment.
        
        Args:
            action_dict: Dictionary mapping agent handles to actions
            current_step: Current environment step
            agent_states: Dictionary mapping agent handles to RailAgentStatus values
            
        Returns:
            Modified action dictionary
        """
        # RailAgentStatus.READY_TO_DEPART = 0
        READY_TO_DEPART = 0
        
        modified_actions = action_dict.copy()
        
        for agent_handle, action in action_dict.items():
            # Only modify agents that are ready to depart but shouldn't yet
            if agent_handle in agent_states:
                if agent_states[agent_handle] == READY_TO_DEPART:
                    if not self.can_depart(agent_handle, current_step):
                        # Force DO_NOTHING to prevent departure
                        modified_actions[agent_handle] = 0
                    else:
                        # Agent can depart - mark as departed
                        self.has_departed[agent_handle] = True
        
        return modified_actions
    
    def get_departure_info(self) -> Dict:
        """Get information about departure schedule."""
        return {
            'n_agents': self.n_agents,
            'spacing': self.departure_spacing,
            'mode': self.spacing_mode,
            'departure_times': self.departure_times.copy(),
            'has_departed': self.has_departed.copy()
        }


def compute_recommended_spacing(
    n_agents: int,
    map_width: int = 40,
    map_height: int = 27,
    n_cities: int = 4
) -> int:
    """
    Compute recommended departure spacing based on environment parameters.
    
    This function estimates how much time agents need to clear initial
    track segments based on map size and agent density.
    
    Args:
        n_agents: Number of agents
        map_width: Width of the map
        map_height: Height of the map
        n_cities: Number of cities (spawn/destination points)
        
    Returns:
        Recommended spacing in steps
    """
    if n_agents <= 1:
        return 0
    
    # Estimate average distance between cities
    map_diagonal = map_width + map_height
    avg_city_distance = map_diagonal / max(2, n_cities)
    
    # Base spacing: give agents time to travel ~1/4 of city distance
    base_spacing = int(avg_city_distance / 4)
    
    # Adjust for agent density
    cells_per_agent = (map_width * map_height) / n_agents
    density_factor = max(0.5, min(2.0, 200 / cells_per_agent))
    
    # Final spacing
    spacing = int(base_spacing * density_factor)
    
    # Clamp to reasonable range
    return max(8, min(40, spacing))


# ============================================================
# Test function
# ============================================================
def test_staggered_departures():
    """Test the staggered departure controller."""
    print("Testing Staggered Departures")
    print("=" * 50)
    
    # Test with 5 agents
    controller = StaggeredDepartureController(
        n_agents=5,
        departure_spacing=25,  # Increased spacing
        spacing_mode='fixed'
    )
    
    print(f"Departure times: {controller.departure_times}")
    print()
    
    # Simulate first 100 steps
    for step in [0, 10, 25, 50, 75, 100]:
        can_depart = {i: controller.can_depart(i, step) for i in range(5)}
        departing = sum(can_depart.values())
        print(f"Step {step:3d}: {departing}/5 agents can depart - {can_depart}")
    
    print()
    print("Recommended spacings for different agent counts:")
    for n in [1, 2, 3, 5, 7, 10]:
        spacing = get_spacing_for_phase(n)
        last_departure = (n - 1) * spacing
        print(f"  {n} agents: spacing={spacing}, last departure at t={last_departure}")


if __name__ == "__main__":
    test_staggered_departures()