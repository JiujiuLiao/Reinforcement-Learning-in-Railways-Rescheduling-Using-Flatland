import os
import sys
import json
import argparse
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs.agent_utils import RailAgentStatus

# Import modules
try:
    from fltlnd.deadlocks import DeadlocksDetector
    import fltlnd.agent as agent_classes
    import fltlnd.obs as obs_classes
    import fltlnd.predict as predictor_classes
    import fltlnd.replay_buffer as memory_classes
    from fltlnd.action_masking import MinimalActionMasker
except ImportError:
    # If running from project root
    from deadlocks import DeadlocksDetector
    import agent as agent_classes
    import obs as obs_classes
    import predict as predictor_classes
    import replay_buffer as memory_classes
    from action_masking import MinimalActionMasker


# Default paths 
DEFAULT_BASE_DIR = ""
DEFAULT_CHECKPOINTS_DIR = "checkpoints"
DEFAULT_ENVIRONMENTS_FILE = "parameters/environments.json"
DEFAULT_SETUP_FILE = "parameters/setup.json"
DEFAULT_RESULTS_DIR = "evaluation_results"

# Test environment configurations (separate from training)
# You can define custom test environments here or use existing ones
TEST_ENVIRONMENTS = {
    "test_small": {
        "n_agents": 3,
        "x_dim": 30,
        "y_dim": 20,
        "n_cities": 3,
        "max_rails_between_cities": 2,
        "max_rails_in_city": 2,
        "grid_mode": False,
        "malfunction_rate": 0.0,
        "malfunction_duration": [0, 0],
        "speed_profiles": [1.0],
    },
    "test_medium": {
        "n_agents": 5,
        "x_dim": 40,
        "y_dim": 27,
        "n_cities": 4,
        "max_rails_between_cities": 3,
        "max_rails_in_city": 3,
        "grid_mode": False,
        "malfunction_rate": 0.0,
        "malfunction_duration": [0, 0],
        "speed_profiles": [1.0],
    },
    "test_large": {
        "n_agents": 10,
        "x_dim": 50,
        "y_dim": 35,
        "n_cities": 5,
        "max_rails_between_cities": 3,
        "max_rails_in_city": 3,
        "grid_mode": False,
        "malfunction_rate": 0.0,
        "malfunction_duration": [0, 0],
        "speed_profiles": [1.0],
    },
    "test_malfunction": {
        "n_agents": 5,
        "x_dim": 40,
        "y_dim": 27,
        "n_cities": 4,
        "max_rails_between_cities": 3,
        "max_rails_in_city": 3,
        "grid_mode": False,
        "malfunction_rate": 0.005,
        "malfunction_duration": [15, 50],
        "speed_profiles": [1.0],
    },
}

# List of all available agents
AVAILABLE_AGENTS = [
    "DQNAgent",
    "DoubleDQNAgent", 
    "DuelingDQNAgent",
    "DDDQNAgent",
    "PPOAgent",
    "RandomAgent",
    "FIFOAgent",
]



def set_all_seeds(seed: int):
    """Set all random seeds for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For complete reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_environment_configs(env_file: str) -> Dict:
    """Load environment configurations from JSON file."""
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            return json.load(f)
    return {}


def load_setup_params(setup_file: str) -> Dict:
    """Load setup parameters from JSON file."""
    if os.path.exists(setup_file):
        with open(setup_file, 'r') as f:
            return json.load(f)
    # Return default parameters if file doesn't exist
    return {
        "sys": {
            "seed": 42,
            "base_dir": "",
            "obs_class": "EnhancedTreeObs",
            "predictor_class": "StochasticPathPredictor"
        },
        "obs": {
            "tree_depth": 3,
            "predictor_depth": 50,
            "radius": 10
        },
        "trn": {
            "memory_size": 100000,
            "batch_size": 128,
            "update_every": 4,
            "learning_rate": 5e-5,
            "tau": 0.001,
            "gamma": 0.99,
            "buffer_min_size": 10000,
            "hidden_sizes": [256, 256],
            "exp_start": 0.0,  # No exploration for evaluation
            "exp_end": 0.0,
            "exp_decay": 1.0,
            "target_update": 1000,
            "soft_update": True,
            "noisy_net": False,
        }
    }


# =============================================================================
# EVALUATION ENVIRONMENT
# =============================================================================

class DeterministicEvalEnv:
    """
    Deterministic evaluation environment wrapper.
    
    Key features:
    - Fixed random seeds for each episode
    - Reproducible environment generation
    - Metrics collection
    """
    
    def __init__(self, env_params: Dict, obs_builder, seed: int = 42):
        self.params = env_params
        self.obs_builder = obs_builder
        self.base_seed = seed
        self.current_episode = 0
        
        self.deadlocks_detector = DeadlocksDetector()
        self.env = None
        
        # Create initial environment
        self._create_env(seed)
    
    def _create_env(self, seed: int):
        """Create a new environment with the given seed."""
        # Set all seeds before environment creation
        set_all_seeds(seed)
        
        # Parse malfunction parameters
        min_mal, max_mal = self.params.get("malfunction_duration", [0, 0])
        malfunction_rate = self.params.get("malfunction_rate", 0.0)
        
        mal_params = mal_gen.MalfunctionParameters(
            malfunction_rate=malfunction_rate,
            min_duration=min_mal,
            max_duration=max_mal
        )
        
        try:
            self.env = RailEnv(
                width=self.params['x_dim'],
                height=self.params['y_dim'],
                rail_generator=sparse_rail_generator(
                    max_num_cities=self.params['n_cities'],
                    seed=seed,
                    grid_mode=self.params.get('grid_mode', False),
                    max_rails_between_cities=self.params['max_rails_between_cities'],
                    max_rails_in_city=self.params['max_rails_in_city']
                ),
                schedule_generator=sparse_schedule_generator(),
                number_of_agents=self.params['n_agents'],
                obs_builder_object=self.obs_builder,
                malfunction_generator=mal_gen.ParamMalfunctionGen(mal_params),
                close_following=False,
                random_seed=seed
            )
        except AttributeError:
            # Fallback for older Flatland versions
            self.env = RailEnv(
                width=self.params['x_dim'],
                height=self.params['y_dim'],
                rail_generator=sparse_rail_generator(
                    max_num_cities=self.params['n_cities'],
                    seed=seed,
                    grid_mode=self.params.get('grid_mode', False),
                    max_rails_between_cities=self.params['max_rails_between_cities'],
                    max_rails_in_city=self.params['max_rails_in_city']
                ),
                schedule_generator=sparse_schedule_generator(),
                number_of_agents=self.params['n_agents'],
                obs_builder_object=self.obs_builder,
                malfunction_generator_and_process_data=mal_gen.malfunction_from_params(mal_params),
                random_seed=seed
            )
    
    def reset(self, episode_idx: int = None):
        """
        Reset environment with deterministic seed based on episode index.
        
        Each episode uses a different but reproducible seed:
        seed = base_seed + episode_idx
        """
        if episode_idx is not None:
            self.current_episode = episode_idx
        
        # Compute deterministic seed for this episode
        episode_seed = self.base_seed + self.current_episode
        set_all_seeds(episode_seed)
        
        # Reset environment
        obs, info = self.env.reset(regenerate_rail=True, regenerate_schedule=True)
        
        # Reset deadlock detector
        self.deadlocks_detector.reset(self.env.get_num_agents())
        info["deadlocks"] = {a: False for a in range(self.env.get_num_agents())}
        
        self.current_episode += 1
        
        return obs, info
    
    def step(self, action_dict: Dict):
        """Execute one step in the environment."""
        next_obs, all_rewards, done, info = self.env.step(action_dict)
        
        # Update deadlock detection
        deadlocks = self.deadlocks_detector.step(self.env)
        info["deadlocks"] = {a: deadlocks[a] for a in range(self.env.get_num_agents())}
        
        return next_obs, all_rewards, done, info
    
    def get_num_agents(self):
        return self.params['n_agents']
    
    def get_agent_handles(self):
        return self.env.get_agent_handles()
    
    @property
    def x_dim(self):
        return self.params['x_dim']
    
    @property
    def y_dim(self):
        return self.params['y_dim']


# =============================================================================
# EVALUATOR CLASS
# =============================================================================

class DeterministicEvaluator:
    """
    Main evaluation class that runs deterministic evaluation of trained agents.
    
    Features:
    - Loads trained models
    - Runs evaluation with epsilon=0 (greedy policy)
    - Collects comprehensive metrics
    - Supports action masking
    """
    
    def __init__(
        self,
        agent_class_name: str,
        env_params: Dict,
        setup_params: Dict,
        checkpoint_path: Optional[str] = None,
        base_dir: str = "",
        use_action_masking: bool = True,
        seed: int = 42,
        verbose: bool = True
    ):
        self.agent_class_name = agent_class_name
        self.env_params = env_params
        self.setup_params = setup_params
        self.checkpoint_path = checkpoint_path
        self.base_dir = base_dir
        self.use_action_masking = use_action_masking
        self.seed = seed
        self.verbose = verbose
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._setup_observation()
        self._setup_environment()
        self._setup_agent()
        
        if use_action_masking:
            self.action_masker = MinimalActionMasker()
        else:
            self.action_masker = None
        
        # Calculate max steps
        self.max_steps = int(4 * 2 * (
            self.env_params['x_dim'] + 
            self.env_params['y_dim'] + 
            (self.env_params['n_agents'] / self.env_params.get('n_cities', 2))
        ))
    
    def _setup_observation(self):
        """Initialize observation wrapper."""
        obs_params = self.setup_params.get('obs', {})
        sys_params = self.setup_params.get('sys', {})
        
        # Get observation class
        obs_class_name = sys_params.get('obs_class', 'EnhancedTreeObs')
        self.obs_class = getattr(obs_classes, obs_class_name)
        
        # Get predictor class
        predictor_class_name = sys_params.get('predictor_class', 'StochasticPathPredictor')
        if predictor_class_name:
            predictor_class = getattr(predictor_classes, predictor_class_name)
            self.predictor = predictor_class(obs_params)
        else:
            self.predictor = None
        
        # Create observation wrapper
        self.obs_wrapper = self.obs_class(obs_params, self.predictor)
    
    def _setup_environment(self):
        """Initialize evaluation environment."""
        self.eval_env = DeterministicEvalEnv(
            env_params=self.env_params,
            obs_builder=self.obs_wrapper.builder,
            seed=self.seed
        )
        self.obs_wrapper.env = self.eval_env
    
    def _setup_agent(self):
        """Initialize and load the agent."""
        # Get agent class
        self.agent_class = getattr(agent_classes, self.agent_class_name)
        
        # Calculate state size
        state_size = self.obs_wrapper.get_state_size()
        if hasattr(self.obs_wrapper, 'n_global_features'):
            state_size += self.obs_wrapper.n_global_features
        else:
            state_size += 3  # Default global features
        
        action_size = 5
        
        # Get training parameters (with exploration disabled)
        trn_params = self.setup_params.get('trn', {}).copy()
        trn_params['exp_start'] = 0.0  # No exploration
        trn_params['exp_end'] = 0.0
        trn_params['exp_decay'] = 1.0
        
        # Get memory class
        if self.agent_class_name == "PPOAgent":
            memory_class = memory_classes.AgentEpisodeBuffer
        else:
            memory_class = memory_classes.ReplayBuffer
        
        # Create agent with exploration DISABLED
        self.agent = self.agent_class(
            state_size=state_size,
            action_size=action_size,
            params=trn_params,
            memory_class=memory_class,
            exploration=False,  # CRITICAL: Disable exploration
            train_best=False,   # Don't auto-load best checkpoint
            base_dir=self.base_dir,
            checkpoint=None     # We'll load manually
        )
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Ensure epsilon is 0
        self.agent.stats['eps_val'] = 0.0
        self.agent._exploration = False
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            # Load from specified path
            if self.checkpoint_path.endswith('.pt'):
                checkpoint_base = self.checkpoint_path[:-3]
            else:
                checkpoint_base = self.checkpoint_path
            self.agent.load(checkpoint_base)
            if self.verbose:
                print(f"Loaded checkpoint from: {self.checkpoint_path}")
        else:
            # Try to load best checkpoint
            best_path = os.path.join(self.base_dir, "checkpoints", str(self.agent))
            if os.path.exists(best_path + ".pt"):
                self.agent.load(best_path)
                if self.verbose:
                    print(f"Loaded best checkpoint from: {best_path}")
            else:
                if self.verbose:
                    print(f"WARNING: No checkpoint found. Using untrained agent.")
    
    def _get_action(self, obs: np.ndarray, agent_handle: int) -> int:
        """
        Get action using greedy policy (epsilon=0).
        
        This method ensures deterministic action selection.
        """
        if self.use_action_masking and self.action_masker:
            # Get action mask
            action_mask = self.action_masker.get_action_mask(
                self.eval_env.env, agent_handle
            )
            
            # Get Q-values from policy network
            state_tensor = torch.as_tensor(
                obs, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.agent._model(state_tensor).cpu().numpy()[0]
            
            # Apply mask
            masked_q = q_values.copy()
            masked_q[action_mask == 0] = -1e9
            
            # Greedy selection (no exploration)
            action = int(np.argmax(masked_q))
        else:
            # Direct greedy action (agent.act with exploration disabled)
            action = self.agent.act(obs)
        
        return action
    
    def run_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Run a single evaluation episode.
        
        Returns:
            Dictionary containing episode metrics
        """
        # Reset environment with deterministic seed
        obs, info = self.eval_env.reset(episode_idx)
        
        # Reset observation wrapper
        if hasattr(self.obs_wrapper, 'reset'):
            self.obs_wrapper.reset()
        
        # Initialize tracking variables
        num_agents = self.eval_env.get_num_agents()
        agent_obs = [None] * num_agents
        action_count = [0] * 5
        total_reward = 0.0
        agent_rewards = defaultdict(float)
        
        # Track deadlocks and completion
        deadlocked_agents = set()
        stuck_agents = set()
        completed_agents = set()
        agent_completion_steps = {}
        
        # Stuck detection
        agent_last_positions = {}
        agent_stationary_steps = defaultdict(int)
        STUCK_THRESHOLD = 50
        
        # Normalize initial observations
        for agent in self.eval_env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = self.obs_wrapper.normalize(
                    obs[agent], self.eval_env, agent
                )
        
        # Run episode
        step_count = 0
        done_all = False
        
        for step in range(self.max_steps):
            step_count += 1
            action_dict = {}
            
            # Select actions for all agents
            for agent in self.eval_env.get_agent_handles():
                agent_obj = self.eval_env.env.agents[agent]
                
                if info['action_required'][agent]:
                    if agent_obs[agent] is not None:
                        action = self._get_action(agent_obs[agent], agent)
                    else:
                        action = 0  # DO_NOTHING if no observation
                    action_count[action] += 1
                else:
                    # Default action when not required
                    if agent_obj.status == RailAgentStatus.ACTIVE:
                        action = 2  # MOVE_FORWARD
                    else:
                        action = 0  # DO_NOTHING
                
                action_dict[agent] = action
            
            # Execute step
            next_obs, all_rewards, done, info = self.eval_env.step(action_dict)
            
            # Update observations and collect rewards
            for agent in self.eval_env.get_agent_handles():
                if next_obs[agent]:
                    agent_obs[agent] = self.obs_wrapper.normalize(
                        next_obs[agent], self.eval_env, agent
                    )
                
                # Accumulate rewards
                reward = all_rewards[agent]
                total_reward += reward
                agent_rewards[agent] += reward
                
                # Check completion
                agent_obj = self.eval_env.env.agents[agent]
                if agent_obj.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                    if agent not in completed_agents:
                        completed_agents.add(agent)
                        agent_completion_steps[agent] = step_count
                
                # Check deadlocks
                if info['deadlocks'].get(agent, False):
                    deadlocked_agents.add(agent)
                
                # Stuck detection (no progress for STUCK_THRESHOLD steps)
                current_pos = agent_obj.position
                if current_pos is not None:
                    if agent_last_positions.get(agent) == current_pos:
                        agent_stationary_steps[agent] += 1
                        if agent_stationary_steps[agent] >= STUCK_THRESHOLD:
                            if agent not in deadlocked_agents:
                                stuck_agents.add(agent)
                    else:
                        agent_stationary_steps[agent] = 0
                    agent_last_positions[agent] = current_pos
            
            # Check if all done
            if done['__all__']:
                done_all = True
                break
        
        # Calculate metrics
        tasks_finished = len(completed_agents)
        completion_rate = tasks_finished / num_agents
        deadlock_rate = len(deadlocked_agents) / num_agents
        stuck_rate = len(stuck_agents) / num_agents
        
        # Normalized reward
        normalized_reward = total_reward / (self.max_steps * num_agents)
        
        # Average completion time for finished agents
        if completed_agents:
            avg_completion_steps = np.mean(list(agent_completion_steps.values()))
        else:
            avg_completion_steps = self.max_steps
        
        # Action distribution
        total_actions = sum(action_count)
        if total_actions > 0:
            action_probs = [c / total_actions for c in action_count]
        else:
            action_probs = [0.2] * 5
        
        return {
            'episode': episode_idx,
            'seed': self.seed + episode_idx,
            'steps': step_count,
            'max_steps': self.max_steps,
            'completion_rate': completion_rate,
            'tasks_finished': tasks_finished,
            'total_agents': num_agents,
            'deadlock_rate': deadlock_rate,
            'stuck_rate': stuck_rate,
            'deadlocked_agents': len(deadlocked_agents),
            'stuck_agents': len(stuck_agents),
            'total_reward': total_reward,
            'normalized_reward': normalized_reward,
            'avg_completion_steps': avg_completion_steps,
            'done_all': done_all,
            'action_counts': action_count,
            'action_probs': action_probs,
        }
    
    def evaluate(self, n_episodes: int = 20) -> Dict[str, Any]:
        """
        Run full evaluation over multiple episodes.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary containing aggregated metrics and per-episode results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f" DETERMINISTIC EVALUATION")
            print(f"{'='*60}")
            print(f" Agent: {self.agent_class_name}")
            print(f" Environment: {self.env_params['n_agents']} agents on "
                  f"{self.env_params['x_dim']}x{self.env_params['y_dim']}")
            print(f" Episodes: {n_episodes}")
            print(f" Base seed: {self.seed}")
            print(f" Action masking: {'ENABLED' if self.use_action_masking else 'DISABLED'}")
            print(f" Exploration (epsilon): 0.0 (DISABLED)")
            print(f"{'='*60}\n")
        
        # Collect per-episode results
        episode_results = []
        
        for ep in range(n_episodes):
            result = self.run_episode(ep)
            episode_results.append(result)
            
            if self.verbose:
                print(f"Episode {ep:3d} | "
                      f"Completion: {result['completion_rate']:.1%} | "
                      f"Deadlocks: {result['deadlocked_agents']} | "
                      f"Stuck: {result['stuck_agents']} | "
                      f"Steps: {result['steps']:4d}/{result['max_steps']} | "
                      f"Reward: {result['normalized_reward']:.4f}")
        
        # Aggregate statistics
        completion_rates = [r['completion_rate'] for r in episode_results]
        deadlock_rates = [r['deadlock_rate'] for r in episode_results]
        stuck_rates = [r['stuck_rate'] for r in episode_results]
        normalized_rewards = [r['normalized_reward'] for r in episode_results]
        steps_list = [r['steps'] for r in episode_results]
        completion_steps = [r['avg_completion_steps'] for r in episode_results]
        
        summary = {
            'agent': self.agent_class_name,
            'env_config': self.env_params,
            'n_episodes': n_episodes,
            'base_seed': self.seed,
            'use_action_masking': self.use_action_masking,
            
            # Completion metrics
            'mean_completion_rate': np.mean(completion_rates),
            'std_completion_rate': np.std(completion_rates),
            'min_completion_rate': np.min(completion_rates),
            'max_completion_rate': np.max(completion_rates),
            
            # Deadlock metrics
            'mean_deadlock_rate': np.mean(deadlock_rates),
            'std_deadlock_rate': np.std(deadlock_rates),
            
            # Stuck metrics
            'mean_stuck_rate': np.mean(stuck_rates),
            'std_stuck_rate': np.std(stuck_rates),
            
            # Reward metrics
            'mean_normalized_reward': np.mean(normalized_rewards),
            'std_normalized_reward': np.std(normalized_rewards),
            'total_reward_sum': sum(r['total_reward'] for r in episode_results),
            
            # Step metrics
            'mean_steps': np.mean(steps_list),
            'std_steps': np.std(steps_list),
            'mean_completion_steps': np.mean(completion_steps),
            
            # Success metrics
            'perfect_episodes': sum(1 for r in episode_results if r['completion_rate'] == 1.0),
            'zero_deadlock_episodes': sum(1 for r in episode_results if r['deadlock_rate'] == 0.0),
            
            # Per-episode results
            'episode_results': episode_results,
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted evaluation summary."""
        print(f"\n{'='*60}")
        print(f" EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f" Agent: {summary['agent']}")
        print(f" Episodes: {summary['n_episodes']}")
        print(f"{'='*60}")
        
        print(f"\n COMPLETION RATE:")
        print(f"   Mean: {summary['mean_completion_rate']:.2%} ± {summary['std_completion_rate']:.2%}")
        print(f"   Range: [{summary['min_completion_rate']:.2%}, {summary['max_completion_rate']:.2%}]")
        print(f"   Perfect episodes (100%): {summary['perfect_episodes']}/{summary['n_episodes']}")
        
        print(f"\n DEADLOCK RATE:")
        print(f"   Mean: {summary['mean_deadlock_rate']:.2%} ± {summary['std_deadlock_rate']:.2%}")
        print(f"   Zero-deadlock episodes: {summary['zero_deadlock_episodes']}/{summary['n_episodes']}")
        
        print(f"\n STUCK RATE:")
        print(f"   Mean: {summary['mean_stuck_rate']:.2%} ± {summary['std_stuck_rate']:.2%}")
        
        print(f"\n NORMALIZED REWARD:")
        print(f"   Mean: {summary['mean_normalized_reward']:.4f} ± {summary['std_normalized_reward']:.4f}")
        
        print(f"\n STEPS:")
        print(f"   Mean steps: {summary['mean_steps']:.1f} ± {summary['std_steps']:.1f}")
        print(f"   Mean completion steps: {summary['mean_completion_steps']:.1f}")
        
        print(f"\n{'='*60}\n")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def save_results(summary: Dict, output_dir: str, agent_name: str, env_name: str):
    """Save evaluation results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{agent_name}_{env_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    summary_converted = convert_types(summary)
    
    with open(filepath, 'w') as f:
        json.dump(summary_converted, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic evaluation of trained Flatland agents"
    )
    
    # Agent arguments
    parser.add_argument(
        '--agent', '-a',
        type=str,
        default='DDDQNAgent',
        choices=AVAILABLE_AGENTS,
        help='Agent class name to evaluate'
    )
    parser.add_argument(
        '--all-agents',
        action='store_true',
        help='Evaluate all available trained agents'
    )
    
    # Environment arguments
    parser.add_argument(
        '--env', '-e',
        type=str,
        default='phase3_five_agents',
        help='Environment name (from environments.json or TEST_ENVIRONMENTS)'
    )
    parser.add_argument(
        '--test-env',
        type=str,
        choices=list(TEST_ENVIRONMENTS.keys()),
        help='Use a predefined test environment'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=20,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Base random seed for reproducibility'
    )
    
    # Model arguments
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to model checkpoint (optional)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=DEFAULT_BASE_DIR,
        help='Base directory for project files'
    )
    
    # Options
    parser.add_argument(
        '--no-action-masking',
        action='store_true',
        help='Disable action masking during evaluation'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Load configuration files
    env_configs = load_environment_configs(
        os.path.join(args.base_dir, DEFAULT_ENVIRONMENTS_FILE)
    )
    setup_params = load_setup_params(
        os.path.join(args.base_dir, DEFAULT_SETUP_FILE)
    )
    
    # Merge test environments
    all_env_configs = {**env_configs, **TEST_ENVIRONMENTS}
    
    # Determine which environment to use
    if args.test_env:
        env_name = args.test_env
        env_params = TEST_ENVIRONMENTS[args.test_env]
    elif args.env in all_env_configs:
        env_name = args.env
        env_params = all_env_configs[args.env]
    else:
        print(f"ERROR: Environment '{args.env}' not found.")
        print(f"Available environments: {list(all_env_configs.keys())}")
        sys.exit(1)
    
    # Determine which agents to evaluate
    if args.all_agents:
        agents_to_evaluate = ['DQNAgent', 'DoubleDQNAgent', 'DuelingDQNAgent', 'DDDQNAgent']
    else:
        agents_to_evaluate = [args.agent]
    
    # Run evaluation for each agent
    all_summaries = {}
    
    for agent_name in agents_to_evaluate:
        if not args.quiet:
            print(f"\n{'#'*60}")
            print(f" Evaluating: {agent_name}")
            print(f"{'#'*60}")
        
        try:
            evaluator = DeterministicEvaluator(
                agent_class_name=agent_name,
                env_params=env_params,
                setup_params=setup_params,
                checkpoint_path=args.checkpoint,
                base_dir=args.base_dir,
                use_action_masking=not args.no_action_masking,
                seed=args.seed,
                verbose=not args.quiet
            )
            
            summary = evaluator.evaluate(n_episodes=args.episodes)
            evaluator.print_summary(summary)
            
            all_summaries[agent_name] = summary
            
            # Save results if requested
            if args.save_results:
                save_results(summary, args.output_dir, agent_name, env_name)
                
        except Exception as e:
            print(f"ERROR evaluating {agent_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison table if multiple agents
    if len(all_summaries) > 1:
        print(f"\n{'='*80}")
        print(f" COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'Agent':<20} {'Completion':>12} {'Deadlock':>12} {'Reward':>12} {'Perfect':>10}")
        print(f"{'-'*80}")
        
        for agent_name, summary in all_summaries.items():
            print(f"{agent_name:<20} "
                  f"{summary['mean_completion_rate']:>11.1%} "
                  f"{summary['mean_deadlock_rate']:>11.1%} "
                  f"{summary['mean_normalized_reward']:>12.4f} "
                  f"{summary['perfect_episodes']:>5}/{summary['n_episodes']:<4}")
        
        print(f"{'='*80}\n")
    
    return all_summaries


if __name__ == "__main__":
    main()