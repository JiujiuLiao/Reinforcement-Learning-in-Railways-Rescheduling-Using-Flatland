"""
Visualization Script for Flatland RL Agents

Renders trained agents in the Flatland environment and saves:
- MP4 videos
- GIF animations  
- PNG frames
- Summary images

Usage:
    python visualize_results.py --agent DQNAgent --env phase3_five_agents --episodes 5 --save_video
    python visualize_results.py --agent DDDQNAgent --env phase3_five_agents --episodes 3 --save_gif
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import imageio
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs.agent_utils import RailAgentStatus
from flatland.utils.rendertools import RenderTool

# Add project root to path
sys.path.insert(0, '.')

from fltlnd.deadlocks import DeadlocksDetector
import fltlnd.agent as agent_classes
import fltlnd.obs as obs_classes
import fltlnd.replay_buffer as memory_classes
import fltlnd.predict as predictor_classes


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_checkpoint_state_size(checkpoint_path: str) -> int:
    """
    Extract the expected state size from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        
    Returns:
        State size (input dimension of first layer)
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Find the first linear layer's weight
        for key, value in state_dict.items():
            if 'weight' in key and len(value.shape) == 2:
                # Shape is [out_features, in_features]
                return value.shape[1]
    except Exception as e:
        print(f"Warning: Could not read checkpoint: {e}")
    
    return None


def get_checkpoint_path(agent_name: str, base_dir: str = "") -> str:
    """Get the checkpoint file path for an agent."""
    # Convert agent class name to checkpoint filename
    # DQNAgent -> dqn-agent.pt
    # DoubleDQNAgent -> double-dqn-agent.pt
    # DuelingDQNAgent -> dueling-dqn-agent.pt
    # DDDQNAgent -> d3qn-agent.pt
    
    name_mapping = {
        'DQNAgent': 'dqn-agent',
        'DoubleDQNAgent': 'double-dqn-agent',
        'DuelingDQNAgent': 'dueling-dqn-agent',
        'DDDQNAgent': 'd3qn-agent',
        'PPOAgent': 'ppo',
    }
    
    checkpoint_name = name_mapping.get(agent_name, agent_name.lower().replace('agent', '-agent'))
    checkpoint_dir = os.path.join(base_dir, "checkpoints") if base_dir else "checkpoints"
    
    return os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")


def create_environment(env_config: dict, obs_builder, seed: int = 42) -> RailEnv:
    """Create a Flatland environment from configuration."""
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
        # Fallback for older Flatland versions
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


# =============================================================================
# OBSERVATION WRAPPER
# =============================================================================

class ObservationManager:
    """
    Manages observation creation and normalization.
    
    Automatically matches observation size to checkpoint requirements.
    """
    
    def __init__(self, setup: dict, checkpoint_state_size: int = None):
        self.setup = setup
        self.checkpoint_state_size = checkpoint_state_size
        
        obs_params = setup['obs']
        predictor = None
        
        # Create predictor if specified
        if setup['sys'].get('predictor_class'):
            predictor_class = getattr(predictor_classes, setup['sys']['predictor_class'])
            predictor = predictor_class(obs_params)
        
        # Determine which observation class to use based on checkpoint
        self._select_observation_class(setup, predictor, obs_params)
        
        self.predictor = predictor
    
    def _select_observation_class(self, setup: dict, predictor, obs_params: dict):
        """Select observation class that matches checkpoint state size."""
        obs_class_name = setup['sys'].get('obs_class', 'TreeObs')
        
        # Try the configured observation class first
        obs_class = getattr(obs_classes, obs_class_name)
        self.obs_wrapper = obs_class(obs_params, predictor)
        
        # Calculate state size for this observation class
        base_size = self.obs_wrapper.get_state_size()
        n_global = getattr(self.obs_wrapper, 'n_global_features', 3)
        current_state_size = base_size + n_global
        
        print(f"Observation class: {obs_class_name}")
        print(f"  Base size: {base_size}, Global features: {n_global}")
        print(f"  Total state size: {current_state_size}")
        
        # If we have a checkpoint, check if sizes match
        if self.checkpoint_state_size is not None:
            print(f"  Checkpoint expects: {self.checkpoint_state_size}")
            
            if current_state_size != self.checkpoint_state_size:
                # Try to find a matching observation class
                print(f"\n  WARNING: State size mismatch!")
                
                # If checkpoint expects fewer features, try TreeObs
                if self.checkpoint_state_size == base_size + 3:
                    print(f"  Switching to TreeObs (3 global features)...")
                    self.obs_wrapper = obs_classes.TreeObs(obs_params, predictor)
                    n_global = 3
                else:
                    print(f"  Could not find matching observation class.")
                    print(f"  Will truncate/pad observations to match checkpoint.")
        
        self.state_size = self.checkpoint_state_size or (base_size + n_global)
        self.n_global_features = n_global
    
    @property
    def builder(self):
        return self.obs_wrapper.builder
    
    def reset(self):
        """Reset observation wrapper state."""
        if hasattr(self.obs_wrapper, 'reset'):
            self.obs_wrapper.reset()
    
    def normalize(self, observation, env, agent_handle: int) -> np.ndarray:
        """
        Normalize observation and ensure correct size for checkpoint.
        """
        # Create a fake env handler for observation normalization
        class EnvProxy:
            def __init__(self, env, config=None):
                self.env = env
                self.x_dim = env.width
                self.y_dim = env.height
            def get_num_agents(self):
                return self.env.get_num_agents()
        
        env_proxy = EnvProxy(env)
        
        # Set env reference for obs_wrapper if needed
        if hasattr(self.obs_wrapper, 'env'):
            self.obs_wrapper.env = env_proxy
        
        # Normalize observation
        normalized = self.obs_wrapper.normalize(observation, env_proxy, agent_handle)
        
        # Adjust size if needed
        if len(normalized) != self.state_size:
            if len(normalized) > self.state_size:
                # Truncate extra features
                normalized = normalized[:self.state_size]
            else:
                # Pad with zeros
                padded = np.zeros(self.state_size)
                padded[:len(normalized)] = normalized
                normalized = padded
        
        return normalized


# =============================================================================
# AGENT LOADER
# =============================================================================

def load_agent(agent_class_name: str, state_size: int, params: dict, 
               base_dir: str = "", checkpoint_path: str = None):
    """
    Load a trained agent from checkpoint.
    
    Args:
        agent_class_name: Name of agent class (e.g., 'DQNAgent')
        state_size: Size of observation state
        params: Training parameters
        base_dir: Base directory for checkpoints
        checkpoint_path: Optional explicit checkpoint path
        
    Returns:
        Loaded agent with exploration disabled
    """
    agent_class = getattr(agent_classes, agent_class_name)
    memory_class = memory_classes.ReplayBuffer
    action_size = 5
    
    # Create agent
    agent = agent_class(
        state_size=state_size,
        action_size=action_size,
        params=params,
        memory_class=memory_class,
        exploration=False,  # Disable exploration for visualization
        train_best=False,   # Don't auto-load, we'll load manually
        base_dir=base_dir,
        checkpoint=None
    )
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(agent_class_name, base_dir)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint_base = checkpoint_path[:-3] if checkpoint_path.endswith('.pt') else checkpoint_path
        agent.load(checkpoint_base)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        print("Using untrained agent!")
    
    # Ensure exploration is disabled
    agent.stats['eps_val'] = 0.0
    agent._exploration = False
    
    return agent


# =============================================================================
# EPISODE RUNNER
# =============================================================================

def run_episode(env: RailEnv, agent, obs_manager: ObservationManager, 
                renderer: RenderTool, deadlock_detector: DeadlocksDetector,
                max_steps: int, capture_frames: bool = True, 
                frame_interval: int = 1) -> tuple:
    """
    Run a single episode with rendering.
    
    Returns:
        frames: List of RGB image arrays
        metrics: Dictionary with episode statistics
    """
    frames = []
    
    # Reset environment and observation manager
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    obs_manager.reset()
    deadlock_detector.reset(env.get_num_agents())
    
    # Initialize observations
    num_agents = env.get_num_agents()
    agent_obs = [None] * num_agents
    
    for handle in range(num_agents):
        if obs[handle] is not None:
            agent_obs[handle] = obs_manager.normalize(obs[handle], env, handle)
    
    # Capture initial frame
    if capture_frames:
        renderer.reset()
        frame = renderer.render_env(
            show=False,
            show_observations=False,
            show_predictions=False,
            return_image=True
        )
        if frame is not None:
            frames.append(frame)
    
    # Run episode
    step = 0
    done_dict = {'__all__': False}
    
    while not done_dict['__all__'] and step < max_steps:
        step += 1
        
        # Get actions
        actions = {}
        for handle in range(num_agents):
            if info['action_required'][handle] and agent_obs[handle] is not None:
                actions[handle] = agent.act(agent_obs[handle])
            else:
                actions[handle] = 0  # DO_NOTHING
        
        # Step environment
        obs, rewards, done_dict, info = env.step(actions)
        
        # Update observations
        for handle in range(num_agents):
            if obs[handle] is not None:
                agent_obs[handle] = obs_manager.normalize(obs[handle], env, handle)
        
        # Update deadlock detection
        deadlocks = deadlock_detector.step(env)
        
        # Capture frame
        if capture_frames and step % frame_interval == 0:
            frame = renderer.render_env(
                show=False,
                show_observations=False,
                show_predictions=False,
                return_image=True
            )
            if frame is not None:
                frames.append(frame)
    
    # Calculate metrics - check ACTUAL completion status, not just done flag
    # An agent is truly "completed" only if it reached its target (DONE or DONE_REMOVED status)
    completed = 0
    for handle in range(num_agents):
        agent_status = env.agents[handle].status
        if agent_status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
            completed += 1
    
    deadlock_count = sum(deadlocks)
    
    metrics = {
        'completion_rate': completed / num_agents,
        'deadlock_rate': deadlock_count / num_agents,
        'steps': step,
        'completed': completed,
        'deadlocks': deadlock_count,
        'total_agents': num_agents,
        'max_steps': max_steps,
        'timed_out': step >= max_steps and not done_dict['__all__'],
    }
    
    return frames, metrics


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_frames_to_disk(frames: list, save_dir: str, episode_idx: int):
    """Save individual frames as PNG files."""
    for i, frame in enumerate(frames):
        filepath = os.path.join(save_dir, f"episode_{episode_idx:03d}_frame_{i:04d}.png")
        imageio.imwrite(filepath, frame)
    print(f"  Saved {len(frames)} frames to {save_dir}")


def save_video(frames: list, filepath: str, fps: int = 10):
    """Save frames as video (MP4 or GIF)."""
    if not frames:
        print("  No frames to save!")
        return
    
    if filepath.endswith('.gif'):
        imageio.mimsave(filepath, frames, fps=fps, loop=0)
    else:
        imageio.mimsave(filepath, frames, fps=fps)
    print(f"  Video saved: {filepath}")


def create_summary_image(frames: list, metrics: dict, filepath: str):
    """Create a summary image showing start, middle, and end states."""
    if len(frames) < 3:
        return
    
    # Select key frames
    first = frames[0]
    middle = frames[len(frames) // 2]
    last = frames[-1]
    
    # Convert to PIL
    img1 = Image.fromarray(first)
    img2 = Image.fromarray(middle)
    img3 = Image.fromarray(last)
    
    # Create combined image
    w, h = img1.width, img1.height
    padding = 10
    header_height = 50
    
    combined = Image.new('RGB', (w * 3 + padding * 4, h + header_height + padding * 2), 'white')
    
    # Paste frames
    y_offset = header_height + padding
    combined.paste(img1, (padding, y_offset))
    combined.paste(img2, (w + padding * 2, y_offset))
    combined.paste(img3, (w * 2 + padding * 3, y_offset))
    
    # Add text
    draw = ImageDraw.Draw(combined)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        label_font = title_font
    
    # Title
    title = f"Completion: {metrics['completion_rate']*100:.0f}% ({metrics['completed']}/{metrics['total_agents']}) | Steps: {metrics['steps']}"
    draw.text((padding, 10), title, fill='black', font=title_font)
    
    # Labels
    labels = ["Start", "Middle", "End"]
    for i, label in enumerate(labels):
        x = padding + i * (w + padding) + w // 2 - 20
        draw.text((x, header_height - 5), label, fill='gray', font=label_font)
    
    combined.save(filepath)
    print(f"  Summary saved: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained Flatland RL agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_results.py --agent DQNAgent --env phase3_five_agents --episodes 5 --save_video
    python visualize_results.py --agent DDDQNAgent --episodes 3 --save_gif --fps 15
    python visualize_results.py --agent DoubleDQNAgent --save_frames --frame_interval 5
        """
    )
    
    # Required arguments
    parser.add_argument("--agent", type=str, default="DQNAgent",
                        help="Agent class name (DQNAgent, DoubleDQNAgent, DuelingDQNAgent, DDDQNAgent)")
    parser.add_argument("--env", type=str, default="phase3_five_agents",
                        help="Environment name from environments.json")
    
    # Episode settings
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to visualize")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Output options
    parser.add_argument("--save_video", action="store_true",
                        help="Save episodes as MP4 videos")
    parser.add_argument("--save_gif", action="store_true",
                        help="Save episodes as GIF animations")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save individual frames as PNG")
    parser.add_argument("--save_summary", action="store_true", default=True,
                        help="Save summary images")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Output directory")
    
    # Video settings
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for video/GIF")
    parser.add_argument("--frame_interval", type=int, default=1,
                        help="Capture every N steps (1=all)")
    
    # Filtering
    parser.add_argument("--min_completion", type=float, default=0.0,
                        help="Minimum completion rate to save (0.0-1.0)")
    parser.add_argument("--max_attempts", type=int, default=50,
                        help="Maximum attempts to find good episodes")
    
    # Advanced
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Custom checkpoint path")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"{args.agent}_{args.env}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("FLATLAND VISUALIZATION")
    print("=" * 60)
    print(f"Agent: {args.agent}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {save_dir}")
    print("=" * 60)
    
    # Load configurations
    with open("parameters/setup.json", "r") as f:
        setup = json.load(f)
    
    with open("parameters/environments.json", "r") as f:
        env_configs = json.load(f)
    
    if args.env not in env_configs:
        print(f"\nERROR: Environment '{args.env}' not found!")
        print(f"Available: {list(env_configs.keys())}")
        return
    
    env_config = env_configs[args.env]
    print(f"\nEnvironment config:")
    print(f"  Agents: {env_config['n_agents']}")
    print(f"  Size: {env_config['x_dim']}x{env_config['y_dim']}")
    print(f"  Cities: {env_config['n_cities']}")
    
    # Detect checkpoint state size
    checkpoint_path = args.checkpoint or get_checkpoint_path(args.agent, setup['sys'].get('base_dir', ''))
    checkpoint_state_size = get_checkpoint_state_size(checkpoint_path)
    
    print(f"\nCheckpoint: {checkpoint_path}")
    if checkpoint_state_size:
        print(f"  State size: {checkpoint_state_size}")
    else:
        print(f"  WARNING: Could not detect state size")
    
    # Create observation manager
    print(f"\nInitializing observations...")
    obs_manager = ObservationManager(setup, checkpoint_state_size)
    
    # Create environment
    env = create_environment(env_config, obs_manager.builder, seed=args.seed)
    
    # Calculate max steps
    max_steps = int(4 * 2 * (env_config['x_dim'] + env_config['y_dim'] + 
                             env_config['n_agents'] / env_config['n_cities']))
    
    # Load agent
    print(f"\nLoading agent...")
    agent = load_agent(
        args.agent,
        obs_manager.state_size,
        setup['trn'],
        setup['sys'].get('base_dir', ''),
        checkpoint_path
    )
    
    # Create components
    deadlock_detector = DeadlocksDetector()
    
    # Run episodes
    print(f"\n{'=' * 60}")
    print("RUNNING EPISODES")
    print('=' * 60)
    
    all_metrics = []
    saved_count = 0
    attempt = 0
    
    while saved_count < args.episodes and attempt < args.max_attempts:
        attempt += 1
        current_seed = args.seed + attempt
        
        # Create fresh environment for this episode
        env = create_environment(env_config, obs_manager.builder, seed=current_seed)
        renderer = RenderTool(env, gl="PILSVG")
        
        print(f"\nEpisode {saved_count + 1}/{args.episodes} (attempt {attempt}, seed={current_seed})")
        
        # Run episode
        frames, metrics = run_episode(
            env, agent, obs_manager, renderer, deadlock_detector,
            max_steps, capture_frames=True, frame_interval=args.frame_interval
        )
        
        print(f"  Completion: {metrics['completion_rate']*100:.1f}% ({metrics['completed']}/{metrics['total_agents']})")
        print(f"  Deadlocks: {metrics['deadlocks']}")
        print(f"  Steps: {metrics['steps']}/{max_steps}")
        print(f"  Frames: {len(frames)}")
        
        # Check minimum completion
        if metrics['completion_rate'] < args.min_completion:
            print(f"  SKIPPED (completion < {args.min_completion*100:.0f}%)")
            continue
        
        # Save outputs
        if args.save_video and frames:
            video_path = os.path.join(save_dir, f"episode_{saved_count:03d}.mp4")
            save_video(frames, video_path, fps=args.fps)
        
        if args.save_gif and frames:
            gif_path = os.path.join(save_dir, f"episode_{saved_count:03d}.gif")
            save_video(frames, gif_path, fps=args.fps)
        
        if args.save_frames and frames:
            save_frames_to_disk(frames, save_dir, saved_count)
        
        if args.save_summary and frames:
            summary_path = os.path.join(save_dir, f"summary_{saved_count:03d}.png")
            create_summary_image(frames, metrics, summary_path)
        
        all_metrics.append(metrics)
        saved_count += 1
    
    # Summary
    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print('=' * 60)
    print(f"Episodes saved: {saved_count}/{args.episodes}")
    
    if all_metrics:
        avg_completion = np.mean([m['completion_rate'] for m in all_metrics])
        avg_deadlocks = np.mean([m['deadlock_rate'] for m in all_metrics])
        avg_steps = np.mean([m['steps'] for m in all_metrics])
        
        print(f"Average completion: {avg_completion*100:.1f}%")
        print(f"Average deadlocks: {avg_deadlocks*100:.1f}%")
        print(f"Average steps: {avg_steps:.1f}")
    
    print(f"Output directory: {save_dir}")
    
    # Save metrics
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "agent": args.agent,
            "environment": args.env,
            "episodes": all_metrics,
            "settings": {
                "seed": args.seed,
                "fps": args.fps,
                "frame_interval": args.frame_interval,
            }
        }, f, indent=2)
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()