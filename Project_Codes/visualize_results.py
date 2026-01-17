"""
Visualization Script for Flatland RL Results

This script loads a trained agent and renders episodes, saving:
1. Individual frames as PNG images
2. A video (MP4/GIF) of the episode

Usage:
    python visualize_results.py --env phase1_two_agents --episodes 5 --save_video
    python visualize_results.py --env phase5_ten_agents --episodes 3 --save_frames
    
Requirements:
    pip install imageio imageio-ffmpeg pillow
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np
import imageio

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs import malfunction_generators as mal_gen
from flatland.utils.rendertools import RenderTool

# Add project root to path
sys.path.insert(0, '.')

from fltlnd.deadlocks import DeadlocksDetector
import fltlnd.agent as agent_classes
import fltlnd.obs as obs_classes
import fltlnd.replay_buffer as memory_classes
import fltlnd.predict as predictor_classes


def create_environment(env_config, obs_builder, seed=42):
    """Create a Flatland environment from config."""
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
    except AttributeError:
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


def load_agent(agent_class_name, state_size, action_size, params, base_dir=""):
    """Load a trained agent from checkpoint."""
    agent_class = getattr(agent_classes, agent_class_name)
    memory_class = memory_classes.ReplayBuffer
    
    agent = agent_class(
        state_size,
        action_size,
        params,
        memory_class,
        exploration=False,  # No exploration during visualization
        train_best=True,    # Load best checkpoint
        base_dir=base_dir,
        checkpoint=None
    )
    
    # Disable exploration completely
    agent.stats["eps_val"] = 0.0
    
    return agent


def run_episode_with_rendering(env, agent, obs_wrapper, renderer, deadlock_detector,
                                max_steps, save_dir=None, episode_idx=0,
                                save_frames=False, frame_interval=1):
    """
    Run one episode and optionally save frames.
    
    Returns:
        frames: list of RGB arrays (for video creation)
        metrics: dict with completion, deadlocks, steps
    """
    frames = []
    
    # Reset environment
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    deadlock_detector.reset(env.get_num_agents())
    
    # Build initial observations
    agent_obs = [None] * env.get_num_agents()
    for handle in range(env.get_num_agents()):
        if obs[handle]:
            agent_obs[handle] = obs_wrapper.normalize(obs[handle], None, handle)
    
    # Render initial state
    renderer.reset()
    frame = renderer.render_env(
        show=False,
        show_observations=False,
        show_predictions=False,
        return_image=True
    )
    if frame is not None:
        frames.append(frame)
        if save_frames and save_dir:
            save_frame(frame, save_dir, episode_idx, 0)
    
    # Run episode
    step = 0
    done_dict = {i: False for i in range(env.get_num_agents())}
    done_dict['__all__'] = False
    
    while not done_dict['__all__'] and step < max_steps:
        step += 1
        
        # Get actions from agent
        actions = {}
        for handle in range(env.get_num_agents()):
            if info['action_required'][handle] and agent_obs[handle] is not None:
                actions[handle] = agent.act(agent_obs[handle])
            else:
                actions[handle] = 0  # DO_NOTHING
        
        # Step environment
        obs, rewards, done_dict, info = env.step(actions)
        
        # Update observations
        for handle in range(env.get_num_agents()):
            if obs[handle]:
                agent_obs[handle] = obs_wrapper.normalize(obs[handle], None, handle)
        
        # Check deadlocks
        deadlocks = deadlock_detector.step(env)
        
        # Render and capture frame
        if step % frame_interval == 0:
            frame = renderer.render_env(
                show=False,
                show_observations=False,
                show_predictions=False,
                return_image=True
            )
            if frame is not None:
                frames.append(frame)
                if save_frames and save_dir:
                    save_frame(frame, save_dir, episode_idx, step)
    
    # Calculate metrics
    completed = sum(1 for i in range(env.get_num_agents()) if done_dict[i])
    deadlock_count = sum(deadlocks)
    
    metrics = {
        'completion_rate': completed / env.get_num_agents(),
        'deadlock_rate': deadlock_count / env.get_num_agents(),
        'steps': step,
        'completed': completed,
        'total_agents': env.get_num_agents()
    }
    
    return frames, metrics


def save_frame(frame, save_dir, episode_idx, step):
    """Save a single frame as PNG."""
    filename = os.path.join(save_dir, f"episode_{episode_idx:03d}_step_{step:04d}.png")
    imageio.imwrite(filename, frame)


def save_video(frames, filepath, fps=10):
    """Save frames as video (MP4 or GIF)."""
    if filepath.endswith('.gif'):
        imageio.mimsave(filepath, frames, fps=fps, loop=0)
    else:
        imageio.mimsave(filepath, frames, fps=fps)
    print(f"Video saved: {filepath}")


def create_visualization_summary(frames, metrics, save_dir, episode_idx):
    """Create a summary image with first, middle, and last frame."""
    if len(frames) < 3:
        return
    
    from PIL import Image, ImageDraw, ImageFont
    
    # Select frames
    first_frame = frames[0]
    middle_frame = frames[len(frames) // 2]
    last_frame = frames[-1]
    
    # Convert to PIL images
    img1 = Image.fromarray(first_frame)
    img2 = Image.fromarray(middle_frame)
    img3 = Image.fromarray(last_frame)
    
    # Create combined image
    width = img1.width
    height = img1.height
    combined = Image.new('RGB', (width * 3 + 40, height + 80), color='white')
    
    # Paste frames
    combined.paste(img1, (10, 60))
    combined.paste(img2, (width + 20, 60))
    combined.paste(img3, (width * 2 + 30, 60))
    
    # Add text
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Title
    title = f"Episode {episode_idx} - Completion: {metrics['completion_rate']*100:.0f}% ({metrics['completed']}/{metrics['total_agents']} agents)"
    draw.text((10, 10), title, fill='black', font=font)
    
    # Labels
    draw.text((10 + width//2 - 30, 40), "Start", fill='black', font=small_font)
    draw.text((width + 20 + width//2 - 30, 40), "Middle", fill='black', font=small_font)
    draw.text((width*2 + 30 + width//2 - 30, 40), "End", fill='black', font=small_font)
    
    # Save
    summary_path = os.path.join(save_dir, f"summary_episode_{episode_idx:03d}.png")
    combined.save(summary_path)
    print(f"Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize trained Flatland agent")
    parser.add_argument("--env", type=str, default="phase1_two_agents",
                        help="Environment name from environments.json")
    parser.add_argument("--agent", type=str, default="DQNAgent",
                        help="Agent class name")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to visualize")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_video", action="store_true",
                        help="Save as video (MP4)")
    parser.add_argument("--save_gif", action="store_true",
                        help="Save as GIF")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save individual frames as PNG")
    parser.add_argument("--save_summary", action="store_true", default=True,
                        help="Save summary image (first/middle/last frame)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for video")
    parser.add_argument("--frame_interval", type=int, default=1,
                        help="Save every N frames (1=all frames)")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Output directory")
    parser.add_argument("--show", action="store_true",
                        help="Show live rendering window")
    parser.add_argument("--select_best", action="store_true", default=True,
                        help="Only save episodes with high completion rate")
    parser.add_argument("--min_completion", type=float, default=0.5,
                        help="Minimum completion rate to save (with --select_best)")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"{args.env}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")
    
    # Load configurations
    with open("parameters/setup.json", "r") as f:
        setup = json.load(f)
    
    with open("parameters/environments.json", "r") as f:
        env_configs = json.load(f)
    
    if args.env not in env_configs:
        print(f"Error: Environment '{args.env}' not found in environments.json")
        print(f"Available: {list(env_configs.keys())}")
        return
    
    env_config = env_configs[args.env]
    print(f"\nEnvironment: {args.env}")
    print(f"  Agents: {env_config['n_agents']}")
    print(f"  Cities: {env_config['n_cities']}")
    print(f"  Map size: {env_config['x_dim']}x{env_config['y_dim']}")
    
    # Create observation wrapper
    obs_params = setup['obs']
    
    # Create predictor if specified
    predictor = None
    if setup['sys'].get('predictor_class'):
        predictor_class = getattr(predictor_classes, setup['sys']['predictor_class'])
        predictor = predictor_class(obs_params)
    
    obs_class = getattr(obs_classes, setup['sys']['obs_class'])
    obs_wrapper = obs_class(obs_params, predictor)
    
    # Create environment
    env = create_environment(env_config, obs_wrapper.builder, seed=args.seed)
    
    # Calculate state size
    state_size = obs_wrapper.get_state_size() + 3  # +3 for global features
    action_size = 5
    
    # Calculate max steps
    max_steps = int(4 * 2 * (env_config['x_dim'] + env_config['y_dim'] + 
                             env_config['n_agents'] / env_config['n_cities']))
    
    # Load trained agent
    print(f"\nLoading agent: {args.agent}")
    agent = load_agent(
        args.agent,
        state_size,
        action_size,
        setup['trn'],
        base_dir=setup['sys']['base_dir']
    )
    print(f"Agent loaded successfully")
    
    # Create renderer
    renderer = RenderTool(env, gl="PILSVG")
    
    # Create deadlock detector
    deadlock_detector = DeadlocksDetector()
    
    # Patch obs_wrapper to work without env_handler
    class FakeEnvHandler:
        def __init__(self, env, config):
            self.env = env
            self.x_dim = config['x_dim']
            self.y_dim = config['y_dim']
        def get_num_agents(self):
            return self.env.get_num_agents()
    
    fake_handler = FakeEnvHandler(env, env_config)
    obs_wrapper.env = fake_handler
    
    # Override normalize method to handle missing env_handler
    original_normalize = obs_wrapper.normalize
    def patched_normalize(observation, env_handler, agent_handle):
        return original_normalize(observation, fake_handler, agent_handle)
    obs_wrapper.normalize = patched_normalize
    
    # Run episodes
    print(f"\nRunning {args.episodes} episodes...")
    
    all_metrics = []
    saved_count = 0
    episode_idx = 0
    attempts = 0
    max_attempts = args.episodes * 10  # Prevent infinite loop
    
    while saved_count < args.episodes and attempts < max_attempts:
        attempts += 1
        
        # Use different seed for each episode
        current_seed = args.seed + attempts
        env = create_environment(env_config, obs_wrapper.builder, seed=current_seed)
        renderer = RenderTool(env, gl="PILSVG")
        fake_handler.env = env
        
        print(f"\n--- Episode {saved_count + 1}/{args.episodes} (attempt {attempts}) ---")
        
        frames, metrics = run_episode_with_rendering(
            env, agent, obs_wrapper, renderer, deadlock_detector,
            max_steps, save_dir, saved_count,
            save_frames=args.save_frames,
            frame_interval=args.frame_interval
        )
        
        print(f"  Completion: {metrics['completion_rate']*100:.1f}% ({metrics['completed']}/{metrics['total_agents']})")
        print(f"  Deadlocks: {metrics['deadlock_rate']*100:.1f}%")
        print(f"  Steps: {metrics['steps']}")
        print(f"  Frames captured: {len(frames)}")
        
        # Check if we should save this episode
        if args.select_best and metrics['completion_rate'] < args.min_completion:
            print(f"  Skipping (completion < {args.min_completion*100:.0f}%)")
            continue
        
        all_metrics.append(metrics)
        
        # Save video
        if args.save_video and frames:
            video_path = os.path.join(save_dir, f"episode_{saved_count:03d}.mp4")
            save_video(frames, video_path, fps=args.fps)
        
        # Save GIF
        if args.save_gif and frames:
            gif_path = os.path.join(save_dir, f"episode_{saved_count:03d}.gif")
            save_video(frames, gif_path, fps=args.fps)
        
        # Save summary
        if args.save_summary and frames:
            create_visualization_summary(frames, metrics, save_dir, saved_count)
        
        saved_count += 1
    
    # Print overall statistics
    print("\n" + "="*50)
    print("VISUALIZATION COMPLETE")
    print("="*50)
    print(f"Episodes saved: {saved_count}")
    if all_metrics:
        avg_completion = np.mean([m['completion_rate'] for m in all_metrics])
        avg_deadlock = np.mean([m['deadlock_rate'] for m in all_metrics])
        print(f"Average completion: {avg_completion*100:.1f}%")
        print(f"Average deadlocks: {avg_deadlock*100:.1f}%")
    print(f"Output directory: {save_dir}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "environment": args.env,
            "agent": args.agent,
            "episodes": all_metrics
        }, f, indent=2)
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()