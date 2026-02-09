"""
Visualization Script with Observation Display
==============================================

Visualizes:
1. Flatland environment with agents moving
2. Agent observation (global features + tree observation summary)
3. Agent status information

Usage:
    python visualize_episode_with_obs.py --agent DDDQNAgent --env phase3_five_agents
    python visualize_episode_with_obs.py --agent DuelingDQNAgent --save_gif
    python visualize_episode_with_obs.py --agent DQNAgent --focus_agent 0
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime
from PIL import Image
import imageio

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs.agent_utils import RailAgentStatus
from flatland.utils.rendertools import RenderTool

sys.path.insert(0, '.')

from fltlnd.deadlocks import DeadlocksDetector
import fltlnd.agent as agent_classes
import fltlnd.obs as obs_classes
import fltlnd.replay_buffer as memory_classes
import fltlnd.predict as predictor_classes


# =============================================================================
# CONFIGURATION
# =============================================================================

ACTION_NAMES = {
    0: "DO_NOTHING",
    1: "MOVE_LEFT", 
    2: "MOVE_FORWARD",
    3: "MOVE_RIGHT",
    4: "STOP_MOVING"
}

STATUS_NAMES = {
    RailAgentStatus.READY_TO_DEPART: "Ready",
    RailAgentStatus.ACTIVE: "Active",
    RailAgentStatus.DONE: "Done",
    RailAgentStatus.DONE_REMOVED: "Completed"
}

GLOBAL_FEATURE_NAMES = [
    "frac_arrived",      # [0]
    "step_fraction",     # [1]
    "speed",             # [2]
    "priority",          # [3]
    "norm_dist_target",  # [4]
    "nearby_density",    # [5]
    "is_blocked",        # [6]
    "head_on_risk",      # [7]
    "head_on_distance",  # [8]
    "agents_ahead",      # [9]
    "agents_behind",     # [10]
    "nearest_dist",      # [11]
    "at_switch",         # [12]
    "should_yield",      # [13]
    "opponent_at_switch",# [14]
    "malfunction",       # [15]
    "steps_stationary",  # [16]
    "conflict_urgency",  # [17]
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_configs(base_dir: str = ""):
    """Load environment and setup configurations."""
    env_path = os.path.join(base_dir, "parameters/environments.json")
    setup_path = os.path.join(base_dir, "parameters/setup.json")
    
    with open(env_path) as f:
        env_configs = json.load(f)
    with open(setup_path) as f:
        setup_config = json.load(f)
    
    return env_configs, setup_config


def create_observation_wrapper(setup_config: dict):
    """Create observation wrapper from config."""
    obs_params = setup_config["obs"]
    sys_params = setup_config["sys"]
    
    # Create predictor
    predictor = None
    if sys_params.get("predictor_class"):
        predictor_class = getattr(predictor_classes, sys_params["predictor_class"])
        predictor = predictor_class(obs_params)
    
    # Create observation wrapper
    obs_class = getattr(obs_classes, sys_params["obs_class"])
    obs_wrapper = obs_class(obs_params, predictor)
    
    return obs_wrapper


def create_environment(env_config: dict, obs_builder, seed: int = 42):
    """Create Flatland environment."""
    min_mal, max_mal = env_config.get("malfunction_duration", [0, 0])
    malfunction_rate = env_config.get("malfunction_rate", 0.0)
    mal_params = mal_gen.MalfunctionParameters(malfunction_rate, min_mal, max_mal)
    
    try:
        env = RailEnv(
            width=env_config["x_dim"],
            height=env_config["y_dim"],
            rail_generator=sparse_rail_generator(
                max_num_cities=env_config["n_cities"],
                seed=seed,
                grid_mode=env_config.get("grid_mode", False),
                max_rails_between_cities=env_config["max_rails_between_cities"],
                max_rails_in_city=env_config["max_rails_in_city"]
            ),
            schedule_generator=sparse_schedule_generator(),
            number_of_agents=env_config["n_agents"],
            obs_builder_object=obs_builder,
            malfunction_generator=mal_gen.ParamMalfunctionGen(mal_params),
            random_seed=seed
        )
    except (AttributeError, TypeError):
        env = RailEnv(
            width=env_config["x_dim"],
            height=env_config["y_dim"],
            rail_generator=sparse_rail_generator(
                max_num_cities=env_config["n_cities"],
                seed=seed,
                grid_mode=env_config.get("grid_mode", False),
                max_rails_between_cities=env_config["max_rails_between_cities"],
                max_rails_in_city=env_config["max_rails_in_city"]
            ),
            schedule_generator=sparse_schedule_generator(),
            number_of_agents=env_config["n_agents"],
            obs_builder_object=obs_builder,
            malfunction_generator_and_process_data=mal_gen.malfunction_from_params(mal_params),
            random_seed=seed
        )
    
    return env


def load_agent(agent_class_name: str, state_size: int, setup_config: dict, base_dir: str = ""):
    """Load trained agent from checkpoint."""
    agent_class = getattr(agent_classes, agent_class_name)
    
    # Use default memory class for loading
    memory_class = getattr(memory_classes, setup_config["sys"].get("memory_class", "ReplayBuffer"))
    
    agent = agent_class(
        state_size=state_size,
        action_size=5,
        params=setup_config["trn"],
        memory_class=memory_class,
        exploration=False,  # Deterministic for visualization
        train_best=True,
        base_dir=base_dir,
        checkpoint=None
    )
    
    return agent


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_global_features(ax, global_features: np.ndarray, agent_handle: int):
    """
    Visualize global coordination features as a bar chart.
    """
    ax.clear()
    
    n_features = min(len(global_features), len(GLOBAL_FEATURE_NAMES))
    features = global_features[-n_features:]  # Last 18 features are global
    
    # Color code by feature type
    colors = []
    for i in range(n_features):
        if i < 3:  # Original features
            colors.append('#2ecc71')  # Green
        elif i < 5:  # Priority features
            colors.append('#3498db')  # Blue
        elif i < 9:  # Conflict features
            colors.append('#e74c3c')  # Red
        elif i < 12:  # Spatial features
            colors.append('#9b59b6')  # Purple
        else:  # Status features
            colors.append('#f39c12')  # Orange
    
    y_pos = np.arange(n_features)
    ax.barh(y_pos, features[:n_features], color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(GLOBAL_FEATURE_NAMES[:n_features], fontsize=8)
    ax.set_xlim([0, 1.1])
    ax.set_xlabel('Value', fontsize=9)
    ax.set_title(f'Agent {agent_handle} Global Features', fontsize=10, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Invert y-axis so first feature is at top
    ax.invert_yaxis()


def visualize_tree_observation(ax, tree_features: np.ndarray):
    """
    Visualize tree observation as a heatmap.
    
    Tree observation has 85 nodes × 11 features = 935 values
    We reshape it to show the structure.
    """
    ax.clear()
    
    # Tree has nodes at each depth: 1 + 4 + 16 + 64 = 85 nodes
    # Each node has ~11 features
    n_features_per_node = 11
    n_nodes = len(tree_features) // n_features_per_node
    
    if n_nodes < 1:
        ax.text(0.5, 0.5, "No tree data", ha='center', va='center')
        return
    
    # Reshape to [nodes, features]
    try:
        tree_matrix = tree_features[:n_nodes * n_features_per_node].reshape(n_nodes, n_features_per_node)
        
        # Show as heatmap (transpose for better visualization)
        im = ax.imshow(tree_matrix.T, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
        
        ax.set_xlabel('Tree Nodes (85 total)', fontsize=9)
        ax.set_ylabel('Features per Node', fontsize=9)
        ax.set_title('Tree Observation Heatmap', fontsize=10, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    except:
        ax.text(0.5, 0.5, "Could not reshape tree data", ha='center', va='center', transform=ax.transAxes)


def visualize_agent_status(ax, env, agent_handle: int, action: int, step: int, deadlocks):
    """
    Show agent status information.
    """
    ax.clear()
    ax.axis('off')
    
    agent = env.agents[agent_handle]
    
    # Handle deadlocks - can be list or dict
    if isinstance(deadlocks, dict):
        is_deadlocked = deadlocks.get(agent_handle, False)
    elif isinstance(deadlocks, list):
        is_deadlocked = deadlocks[agent_handle] if agent_handle < len(deadlocks) else False
    else:
        is_deadlocked = False
    
    status_text = []
    status_text.append(f"═══════════════════════════")
    status_text.append(f"  AGENT {agent_handle} STATUS")
    status_text.append(f"═══════════════════════════")
    status_text.append(f"")
    status_text.append(f"Step: {step}")
    status_text.append(f"Status: {STATUS_NAMES.get(agent.status, str(agent.status))}")
    status_text.append(f"Position: {agent.position}")
    status_text.append(f"Target: {agent.target}")
    status_text.append(f"Direction: {agent.direction}")
    status_text.append(f"")
    status_text.append(f"Action: {ACTION_NAMES.get(action, str(action))}")
    status_text.append(f"Deadlock: {'YES ⚠️' if is_deadlocked else 'No'}")
    
    if agent.position and agent.target:
        dist = abs(agent.position[0] - agent.target[0]) + abs(agent.position[1] - agent.target[1])
        status_text.append(f"Distance to target: {dist}")
    
    # Join and display
    text = '\n'.join(status_text)
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def create_combined_frame(env_frame, obs_data: dict, env, focus_agent: int, 
                          action: int, step: int, deadlocks: dict) -> np.ndarray:
    """
    Create a combined frame with environment and observation visualization.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1])
    
    # Environment view (left, spans 2 rows)
    ax_env = fig.add_subplot(gs[:, 0])
    ax_env.imshow(env_frame)
    ax_env.set_title(f'Flatland Environment - Step {step}', fontsize=12, fontweight='bold')
    ax_env.axis('off')
    
    # Global features (top right)
    ax_global = fig.add_subplot(gs[0, 1])
    if obs_data.get('normalized') is not None:
        global_features = obs_data['normalized'][-18:]  # Last 18 are global
        visualize_global_features(ax_global, global_features, focus_agent)
    else:
        ax_global.text(0.5, 0.5, "No observation", ha='center', va='center')
        ax_global.axis('off')
    
    # Agent status (top far right)
    ax_status = fig.add_subplot(gs[0, 2])
    visualize_agent_status(ax_status, env, focus_agent, action, step, deadlocks)
    
    # Tree observation heatmap (bottom right, spans 2 columns)
    ax_tree = fig.add_subplot(gs[1, 1:])
    if obs_data.get('normalized') is not None:
        tree_features = obs_data['normalized'][:-18]  # All except last 18
        visualize_tree_observation(ax_tree, tree_features)
    else:
        ax_tree.text(0.5, 0.5, "No tree data", ha='center', va='center')
        ax_tree.axis('off')
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return frame


# =============================================================================
# MAIN VISUALIZATION LOOP
# =============================================================================

def run_visualization(agent_class_name: str, env_name: str, focus_agent: int = 0,
                      seed: int = 42, save_gif: bool = True, save_frames: bool = False,
                      output_dir: str = "visualizations", max_steps: int = None):
    """
    Run one episode with visualization.
    """
    print(f"\n{'='*60}")
    print(f"  FLATLAND VISUALIZATION WITH OBSERVATION")
    print(f"{'='*60}")
    print(f"Agent: {agent_class_name}")
    print(f"Environment: {env_name}")
    print(f"Focus Agent: {focus_agent}")
    print(f"Seed: {seed}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configurations
    env_configs, setup_config = load_configs()
    env_config = env_configs[env_name]
    
    # =========================================================================
    # VALIDATE focus_agent - ADD THIS CHECK
    # =========================================================================
    n_agents = env_config["n_agents"]
    if focus_agent < 0 or focus_agent >= n_agents:
        print(f"WARNING: focus_agent={focus_agent} is invalid!")
        print(f"Environment '{env_name}' has {n_agents} agents (valid indices: 0-{n_agents-1})")
        focus_agent = 0
        print(f"Setting focus_agent to {focus_agent}")
    # =========================================================================
    
    # Create observation wrapper
    obs_wrapper = create_observation_wrapper(setup_config)
    
    # Create environment
    env = create_environment(env_config, obs_wrapper.builder, seed)
    renderer = RenderTool(env)
    deadlock_detector = DeadlocksDetector()
    
    # Calculate state size
    tree_size = obs_wrapper.get_state_size()
    global_size = getattr(obs_wrapper, 'n_global_features', 18)
    state_size = tree_size + global_size
    
    print(f"State size: {state_size} (tree: {tree_size}, global: {global_size})")
    
    # Load agent
    agent = load_agent(agent_class_name, state_size, setup_config)
    
    # Disable exploration for deterministic visualization
    agent.stats['eps_val'] = 0.0
    
    # Calculate max steps
    if max_steps is None:
        max_steps = int(4 * 2 * (env_config['x_dim'] + env_config['y_dim'] + 
                                  env_config['n_agents'] / env_config['n_cities']))
    
    # Reset environment
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    deadlock_detector.reset(env.get_num_agents())
    
    # Attach env to obs_wrapper for global features
    class EnvHandler:
        def __init__(self, env, x_dim, y_dim):
            self.env = env
            self.x_dim = x_dim
            self.y_dim = y_dim
        def get_num_agents(self):
            return env.get_num_agents()
        def get_agents_handle(self):
            return env.get_agent_handles()
    
    env_handler = EnvHandler(env, env_config['x_dim'], env_config['y_dim'])
    if hasattr(obs_wrapper, 'reset'):
        obs_wrapper.reset()
    
    # Normalize initial observations
    agent_obs = {}
    for agent_handle in env.get_agent_handles():
        if obs[agent_handle] is not None:
            agent_obs[agent_handle] = obs_wrapper.normalize(obs[agent_handle], env_handler, agent_handle)
    
    frames = []
    
    print(f"\nRunning episode (max {max_steps} steps)...")
    
    for step in range(max_steps):
        # Get actions for all agents
        action_dict = {}
        for agent_handle in env.get_agent_handles():
            agent_obj = env.agents[agent_handle]
            
            if info['action_required'][agent_handle] and agent_obs.get(agent_handle) is not None:
                action = agent.act(agent_obs[agent_handle])
            else:
                if agent_obj.status == RailAgentStatus.ACTIVE:
                    action = 2  # FORWARD
                else:
                    action = 0  # DO_NOTHING
            
            action_dict[agent_handle] = action
        
        # Render environment
        renderer.reset()
        env_frame = renderer.render_env(
            show=False,
            show_observations=False,
            show_predictions=False,
            return_image=True
        )
        
        # Get deadlocks
        deadlocks = deadlock_detector.step(env)
        
        # Prepare observation data for focused agent
        obs_data = {}
        if focus_agent in agent_obs and agent_obs[focus_agent] is not None:
            obs_data['normalized'] = agent_obs[focus_agent]
        
        # Create combined frame
        combined_frame = create_combined_frame(
            env_frame=env_frame,
            obs_data=obs_data,
            env=env,
            focus_agent=focus_agent,
            action=action_dict.get(focus_agent, 0),
            step=step,
            deadlocks=deadlocks
        )
        
        frames.append(combined_frame)
        
        # Save individual frame if requested
        if save_frames:
            frame_path = os.path.join(output_dir, f"frame_{step:04d}.png")
            Image.fromarray(combined_frame).save(frame_path)
        
        # Step environment
        next_obs, rewards, done, info = env.step(action_dict)
        
        # Update observations
        for agent_handle in env.get_agent_handles():
            if next_obs[agent_handle] is not None:
                agent_obs[agent_handle] = obs_wrapper.normalize(next_obs[agent_handle], env_handler, agent_handle)
        
        # Print progress
        if step % 50 == 0:
            completed = sum(1 for a in env.agents if a.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED])
            print(f"  Step {step}: {completed}/{env.get_num_agents()} completed")
        
        # Check if done
        if done['__all__']:
            print(f"  Episode finished at step {step}")
            break
    
    # Calculate final statistics
    completed = sum(1 for a in env.agents if a.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED])
    print(f"\nFinal: {completed}/{env.get_num_agents()} agents completed")
    
    # Save GIF
    if save_gif and frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(output_dir, f"episode_{agent_class_name}_{env_name}_{timestamp}.gif")
        
        print(f"\nSaving GIF to: {gif_path}")
        imageio.mimsave(gif_path, frames, fps=10, loop=0)
        print(f"Saved {len(frames)} frames")
    
    return frames


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Flatland episode with observation display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_episode_with_obs.py --agent DDDQNAgent --env phase3_five_agents
  python visualize_episode_with_obs.py --agent DuelingDQNAgent --focus_agent 2
  python visualize_episode_with_obs.py --agent DQNAgent --save_frames
        """
    )
    
    parser.add_argument("--agent", type=str, default="DDDQNAgent",
                        help="Agent class name")
    parser.add_argument("--env", type=str, default="phase3_five_agents",
                        help="Environment name")
    parser.add_argument("--focus_agent", type=int, default=0,
                        help="Which agent to focus observation visualization on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_gif", action="store_true", default=True,
                        help="Save as GIF")
    parser.add_argument("--no_gif", action="store_true",
                        help="Don't save GIF")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save individual frames")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Output directory")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum steps per episode")
    
    args = parser.parse_args()
    
    run_visualization(
        agent_class_name=args.agent,
        env_name=args.env,
        focus_agent=args.focus_agent,
        seed=args.seed,
        save_gif=not args.no_gif,
        save_frames=args.save_frames,
        output_dir=args.output_dir,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()