#!/usr/bin/env python3
"""
Enhanced Deterministic Evaluation Script for Flatland RL Agents
===============================================================

This script provides thesis-quality evaluation with:
1. Deterministic evaluation (epsilon=0) for reproducible results
2. Statistical analysis with confidence intervals
3. Comparison tables for multiple agents
4. Publication-ready output formats (CSV, LaTeX tables)
5. Visualization of results

Based on the original evaluate.py with enhancements for academic reporting.

Usage Examples:
--------------
# Evaluate single agent (D3QN) with 100 episodes
python evaluate_enhanced.py --agent DDDQNAgent --episodes 100 --save-results

# Evaluate all DQN variants and compare
python evaluate_enhanced.py --all-agents --episodes 100 --save-results

# Evaluate on specific environment
python evaluate_enhanced.py --agent DDDQNAgent --env phase3_five_agents --episodes 100

# Quick test with 20 episodes
python evaluate_enhanced.py --agent DDDQNAgent --episodes 20

# Generate LaTeX table for thesis
python evaluate_enhanced.py --all-agents --episodes 100 --latex

Author: Enhanced for thesis evaluation
"""

import os
import sys
import json
import argparse
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from scipy import stats

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization disabled.")

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs import malfunction_generators as mal_gen
from flatland.envs.agent_utils import RailAgentStatus

# Import project modules
try:
    from fltlnd.deadlocks import DeadlocksDetector
    import fltlnd.agent as agent_classes
    import fltlnd.obs as obs_classes
    import fltlnd.predict as predictor_classes
    import fltlnd.replay_buffer as memory_classes
    from fltlnd.action_masking import MinimalActionMasker
except ImportError:
    from deadlocks import DeadlocksDetector
    import agent as agent_classes
    import obs as obs_classes
    import predict as predictor_classes
    import replay_buffer as memory_classes
    from action_masking import MinimalActionMasker


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BASE_DIR = ""
DEFAULT_CHECKPOINTS_DIR = "checkpoints"
DEFAULT_ENVIRONMENTS_FILE = "parameters/environments.json"
DEFAULT_SETUP_FILE = "parameters/setup.json"
DEFAULT_RESULTS_DIR = "evaluation_results"

# Agents to evaluate (in order for comparison tables)
AGENT_ORDER = [
    "DQNAgent",
    "DoubleDQNAgent",
    "DuelingDQNAgent",
    "DDDQNAgent",
]

AGENT_DISPLAY_NAMES = {
    "DQNAgent": "DQN",
    "DoubleDQNAgent": "Double DQN",
    "DuelingDQNAgent": "Dueling DQN",
    "DDDQNAgent": "D3QN",
    "RandomAgent": "Random",
    "FIFOAgent": "FIFO",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for the mean.
    
    Returns:
        (lower_bound, upper_bound) tuple
    """
    n = len(data)
    if n < 2:
        return (np.mean(data), np.mean(data))
    
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error of mean
    
    # t-value for confidence interval
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_val * se
    
    return (mean - margin, mean + margin)


def format_ci(mean: float, ci_low: float, ci_high: float, is_percent: bool = True) -> str:
    """Format mean with confidence interval for display."""
    if is_percent:
        return f"{mean*100:.1f}% [{ci_low*100:.1f}, {ci_high*100:.1f}]"
    else:
        return f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"


def load_json_file(filepath: str) -> Dict:
    """Load JSON file with error handling."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


# =============================================================================
# EVALUATION ENVIRONMENT
# =============================================================================

class DeterministicEvalEnv:
    """Deterministic evaluation environment wrapper."""
    
    def __init__(self, env_params: Dict, obs_builder, seed: int = 42):
        self.params = env_params
        self.obs_builder = obs_builder
        self.base_seed = seed
        self.current_episode = 0
        self.deadlocks_detector = DeadlocksDetector()
        self.env = None
        self._create_env(seed)
    
    def _create_env(self, seed: int):
        """Create environment with given seed."""
        set_all_seeds(seed)
        
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
        except (AttributeError, TypeError):
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
        """Reset with deterministic seed."""
        if episode_idx is not None:
            self.current_episode = episode_idx
        
        episode_seed = self.base_seed + self.current_episode
        set_all_seeds(episode_seed)
        
        obs, info = self.env.reset(regenerate_rail=True, regenerate_schedule=True)
        self.deadlocks_detector.reset(self.env.get_num_agents())
        info["deadlocks"] = {a: False for a in range(self.env.get_num_agents())}
        
        self.current_episode += 1
        return obs, info
    
    def step(self, action_dict: Dict):
        """Execute step."""
        next_obs, all_rewards, done, info = self.env.step(action_dict)
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
# MAIN EVALUATOR CLASS
# =============================================================================

class EnhancedEvaluator:
    """
    Enhanced evaluation class with thesis-quality metrics.
    
    Features:
    - Deterministic evaluation (epsilon=0)
    - Confidence intervals
    - Statistical significance tests
    - Multiple output formats
    """
    
    def __init__(
        self,
        agent_class_name: str,
        env_params: Dict,
        setup_params: Dict,
        checkpoint_path: Optional[str] = None,
        base_dir: str = "",
        use_action_masking: bool = False,  # Disabled by default (matches training)
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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._setup_observation()
        self._setup_environment()
        self._setup_agent()
        
        if use_action_masking:
            self.action_masker = MinimalActionMasker()
        else:
            self.action_masker = None
        
        # Calculate max steps (same formula as training)
        self.max_steps = int(4 * 2 * (
            self.env_params['x_dim'] + 
            self.env_params['y_dim'] + 
            (self.env_params['n_agents'] / self.env_params.get('n_cities', 2))
        ))
    
    def _setup_observation(self):
        """Initialize observation wrapper."""
        obs_params = self.setup_params.get('obs', {})
        sys_params = self.setup_params.get('sys', {})
        
        obs_class_name = sys_params.get('obs_class', 'EnhancedTreeObs')
        self.obs_class = getattr(obs_classes, obs_class_name)
        
        predictor_class_name = sys_params.get('predictor_class', 'StochasticPathPredictor')
        if predictor_class_name:
            predictor_class = getattr(predictor_classes, predictor_class_name)
            self.predictor = predictor_class(obs_params)
        else:
            self.predictor = None
        
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
        """Initialize and load agent."""
        self.agent_class = getattr(agent_classes, self.agent_class_name)
        
        # Calculate state size
        state_size = self.obs_wrapper.get_state_size()
        if hasattr(self.obs_wrapper, 'n_global_features'):
            state_size += self.obs_wrapper.n_global_features
        else:
            state_size += 3
        
        action_size = 5
        
        # Disable exploration for evaluation
        trn_params = self.setup_params.get('trn', {}).copy()
        trn_params['exp_start'] = 0.0
        trn_params['exp_end'] = 0.0
        trn_params['exp_decay'] = 1.0
        
        # Get memory class
        if self.agent_class_name == "PPOAgent":
            memory_class = memory_classes.AgentEpisodeBuffer
        else:
            memory_class = getattr(memory_classes, 'ReplayBuffer', memory_classes.PrioritizedBuffer)
        
        # Create agent
        self.agent = self.agent_class(
            state_size=state_size,
            action_size=action_size,
            params=trn_params,
            memory_class=memory_class,
            exploration=False,
            train_best=False,
            base_dir=self.base_dir,
            checkpoint=None
        )
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Force epsilon to 0
        self.agent.stats['eps_val'] = 0.0
        self.agent._exploration = False
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        loaded = False
        
        # Try explicit checkpoint path first
        if self.checkpoint_path:
            path = self.checkpoint_path
            if path.endswith('.pt'):
                path = path[:-3]
            if os.path.exists(path + ".pt"):
                self.agent.load(path)
                if self.verbose:
                    print(f"✓ Loaded checkpoint: {path}.pt")
                loaded = True
        
        # Try best checkpoint
        if not loaded:
            best_path = os.path.join(self.base_dir, "checkpoints", str(self.agent))
            if os.path.exists(best_path + ".pt"):
                self.agent.load(best_path)
                if self.verbose:
                    print(f"✓ Loaded best checkpoint: {best_path}.pt")
                loaded = True
        
        if not loaded and self.verbose:
            print(f"⚠ WARNING: No checkpoint found for {self.agent_class_name}")
    
    def _get_action(self, obs: np.ndarray, agent_handle: int) -> int:
        """Get greedy action (epsilon=0)."""
        return self.agent.act(obs)
    
    def run_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Run single evaluation episode."""
        obs, info = self.eval_env.reset(episode_idx)
        
        if hasattr(self.obs_wrapper, 'reset'):
            self.obs_wrapper.reset()
        
        num_agents = self.eval_env.get_num_agents()
        agent_obs = [None] * num_agents
        action_count = [0] * 5
        total_reward = 0.0
        
        deadlocked_agents = set()
        stuck_agents = set()
        completed_agents = set()
        agent_completion_steps = {}
        
        agent_last_positions = {}
        agent_stationary_steps = defaultdict(int)
        STUCK_THRESHOLD = 50
        
        # Normalize initial observations
        for agent in self.eval_env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = self.obs_wrapper.normalize(
                    obs[agent], self.eval_env, agent
                )
        
        step_count = 0
        
        for step in range(self.max_steps):
            step_count += 1
            action_dict = {}
            
            for agent in self.eval_env.get_agent_handles():
                agent_obj = self.eval_env.env.agents[agent]
                
                if info['action_required'][agent]:
                    if agent_obs[agent] is not None:
                        action = self._get_action(agent_obs[agent], agent)
                    else:
                        action = 0
                    action_count[action] += 1
                else:
                    action = 0
                
                action_dict[agent] = action
            
            next_obs, all_rewards, done, info = self.eval_env.step(action_dict)
            
            for agent in self.eval_env.get_agent_handles():
                if next_obs[agent]:
                    agent_obs[agent] = self.obs_wrapper.normalize(
                        next_obs[agent], self.eval_env, agent
                    )
                
                total_reward += all_rewards[agent]
                
                agent_obj = self.eval_env.env.agents[agent]
                if agent_obj.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:
                    if agent not in completed_agents:
                        completed_agents.add(agent)
                        agent_completion_steps[agent] = step_count
                
                if info['deadlocks'].get(agent, False):
                    deadlocked_agents.add(agent)
                
                # Stuck detection
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
            
            if done['__all__']:
                break
        
        completion_rate = len(completed_agents) / num_agents
        deadlock_rate = len(deadlocked_agents) / num_agents
        stuck_rate = len(stuck_agents) / num_agents
        normalized_reward = total_reward / (self.max_steps * num_agents)
        
        avg_completion_steps = (
            np.mean(list(agent_completion_steps.values())) 
            if completed_agents else self.max_steps
        )
        
        total_actions = sum(action_count)
        action_probs = [c / total_actions if total_actions > 0 else 0.2 for c in action_count]
        
        return {
            'episode': episode_idx,
            'seed': self.seed + episode_idx,
            'steps': step_count,
            'completion_rate': completion_rate,
            'deadlock_rate': deadlock_rate,
            'stuck_rate': stuck_rate,
            'normalized_reward': normalized_reward,
            'avg_completion_steps': avg_completion_steps,
            'action_probs': action_probs,
            'tasks_finished': len(completed_agents),
            'total_agents': num_agents,
        }
    
    def evaluate(self, n_episodes: int = 100) -> Dict[str, Any]:
        """Run full evaluation with statistical analysis."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f" DETERMINISTIC EVALUATION")
            print(f"{'='*60}")
            print(f" Agent: {AGENT_DISPLAY_NAMES.get(self.agent_class_name, self.agent_class_name)}")
            print(f" Environment: {self.env_params['n_agents']} agents on "
                  f"{self.env_params['x_dim']}x{self.env_params['y_dim']}")
            print(f" Episodes: {n_episodes}")
            print(f" Seed: {self.seed}")
            print(f" Action masking: {'ON' if self.use_action_masking else 'OFF'}")
            print(f" Exploration: OFF (ε=0)")
            print(f"{'='*60}\n")
        
        episode_results = []
        start_time = time.time()
        
        for ep in range(n_episodes):
            result = self.run_episode(ep)
            episode_results.append(result)
            
            if self.verbose:
                print(f"  Episode {ep+1:3d}/{n_episodes} | "
                      f"Completion: {result['completion_rate']:.0%} | "
                      f"Deadlock: {result['deadlock_rate']:.0%} | "
                      f"Steps: {result['steps']:4d}")
        
        eval_time = time.time() - start_time
        
        # Extract metrics
        completion_rates = [r['completion_rate'] for r in episode_results]
        deadlock_rates = [r['deadlock_rate'] for r in episode_results]
        stuck_rates = [r['stuck_rate'] for r in episode_results]
        normalized_rewards = [r['normalized_reward'] for r in episode_results]
        steps_list = [r['steps'] for r in episode_results]
        
        # Calculate confidence intervals (95%)
        comp_ci = calculate_confidence_interval(completion_rates)
        dead_ci = calculate_confidence_interval(deadlock_rates)
        reward_ci = calculate_confidence_interval(normalized_rewards)
        
        summary = {
            'agent': self.agent_class_name,
            'agent_display': AGENT_DISPLAY_NAMES.get(self.agent_class_name, self.agent_class_name),
            'env_config': self.env_params,
            'n_episodes': n_episodes,
            'base_seed': self.seed,
            'evaluation_time_seconds': eval_time,
            
            # Completion metrics
            'mean_completion_rate': np.mean(completion_rates),
            'std_completion_rate': np.std(completion_rates),
            'ci_completion_rate': comp_ci,
            'min_completion_rate': np.min(completion_rates),
            'max_completion_rate': np.max(completion_rates),
            
            # Deadlock metrics
            'mean_deadlock_rate': np.mean(deadlock_rates),
            'std_deadlock_rate': np.std(deadlock_rates),
            'ci_deadlock_rate': dead_ci,
            
            # Stuck metrics
            'mean_stuck_rate': np.mean(stuck_rates),
            'std_stuck_rate': np.std(stuck_rates),
            
            # Reward metrics
            'mean_normalized_reward': np.mean(normalized_rewards),
            'std_normalized_reward': np.std(normalized_rewards),
            'ci_normalized_reward': reward_ci,
            
            # Step metrics
            'mean_steps': np.mean(steps_list),
            'std_steps': np.std(steps_list),
            
            # Success counts
            'perfect_episodes': sum(1 for r in episode_results if r['completion_rate'] == 1.0),
            'zero_deadlock_episodes': sum(1 for r in episode_results if r['deadlock_rate'] == 0.0),
            
            # Raw results
            'episode_results': episode_results,
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary."""
        print(f"\n{'='*60}")
        print(f" EVALUATION RESULTS: {summary['agent_display']}")
        print(f"{'='*60}")
        print(f" Episodes evaluated: {summary['n_episodes']}")
        print(f" Time: {summary['evaluation_time_seconds']:.1f}s")
        print(f"{'='*60}")
        
        print(f"\n COMPLETION RATE:")
        print(f"   Mean ± Std: {summary['mean_completion_rate']:.1%} ± {summary['std_completion_rate']:.1%}")
        print(f"   95% CI: [{summary['ci_completion_rate'][0]:.1%}, {summary['ci_completion_rate'][1]:.1%}]")
        print(f"   Range: [{summary['min_completion_rate']:.0%}, {summary['max_completion_rate']:.0%}]")
        print(f"   Perfect (100%): {summary['perfect_episodes']}/{summary['n_episodes']}")
        
        print(f"\n DEADLOCK RATE:")
        print(f"   Mean ± Std: {summary['mean_deadlock_rate']:.1%} ± {summary['std_deadlock_rate']:.1%}")
        print(f"   95% CI: [{summary['ci_deadlock_rate'][0]:.1%}, {summary['ci_deadlock_rate'][1]:.1%}]")
        print(f"   Zero-deadlock: {summary['zero_deadlock_episodes']}/{summary['n_episodes']}")
        
        print(f"\n NORMALIZED REWARD:")
        print(f"   Mean ± Std: {summary['mean_normalized_reward']:.4f} ± {summary['std_normalized_reward']:.4f}")
        
        print(f"\n{'='*60}\n")


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_comparison_table(summaries: Dict[str, Dict], output_dir: str = None):
    """Generate comparison table for multiple agents."""
    print(f"\n{'='*90}")
    print(f" COMPARISON TABLE - All Agents")
    print(f"{'='*90}")
    print(f"{'Agent':<15} {'Completion':>18} {'Deadlock':>18} {'Reward':>15} {'Perfect':>10}")
    print(f"{'-'*90}")
    
    for agent_name in AGENT_ORDER:
        if agent_name not in summaries:
            continue
        s = summaries[agent_name]
        display_name = AGENT_DISPLAY_NAMES.get(agent_name, agent_name)
        
        comp_str = f"{s['mean_completion_rate']*100:.1f}% ± {s['std_completion_rate']*100:.1f}%"
        dead_str = f"{s['mean_deadlock_rate']*100:.1f}% ± {s['std_deadlock_rate']*100:.1f}%"
        reward_str = f"{s['mean_normalized_reward']:.4f}"
        perfect_str = f"{s['perfect_episodes']}/{s['n_episodes']}"
        
        print(f"{display_name:<15} {comp_str:>18} {dead_str:>18} {reward_str:>15} {perfect_str:>10}")
    
    print(f"{'='*90}\n")


def generate_latex_table(summaries: Dict[str, Dict], output_file: str = None) -> str:
    """Generate LaTeX table for thesis."""
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Evaluation Results for DQN Variants on 5-Agent Scenario}")
    latex.append(r"\label{tab:evaluation_results}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"Agent & Completion Rate & Deadlock Rate & Reward & Perfect Episodes \\")
    latex.append(r"\midrule")
    
    for agent_name in AGENT_ORDER:
        if agent_name not in summaries:
            continue
        s = summaries[agent_name]
        display_name = AGENT_DISPLAY_NAMES.get(agent_name, agent_name)
        
        comp = f"${s['mean_completion_rate']*100:.1f}\\% \\pm {s['std_completion_rate']*100:.1f}\\%$"
        dead = f"${s['mean_deadlock_rate']*100:.1f}\\% \\pm {s['std_deadlock_rate']*100:.1f}\\%$"
        reward = f"${s['mean_normalized_reward']:.4f}$"
        perfect = f"${s['perfect_episodes']}/{s['n_episodes']}$"
        
        latex.append(f"{display_name} & {comp} & {dead} & {reward} & {perfect} \\\\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    latex_str = "\n".join(latex)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {output_file}")
    
    print("\n" + latex_str + "\n")
    return latex_str


def generate_csv_results(summaries: Dict[str, Dict], output_file: str):
    """Generate CSV file with results."""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Agent', 'Completion_Mean', 'Completion_Std', 'Completion_CI_Low', 'Completion_CI_High',
            'Deadlock_Mean', 'Deadlock_Std', 'Deadlock_CI_Low', 'Deadlock_CI_High',
            'Reward_Mean', 'Reward_Std', 'Perfect_Episodes', 'Zero_Deadlock_Episodes', 'N_Episodes'
        ])
        
        for agent_name in AGENT_ORDER:
            if agent_name not in summaries:
                continue
            s = summaries[agent_name]
            
            writer.writerow([
                AGENT_DISPLAY_NAMES.get(agent_name, agent_name),
                f"{s['mean_completion_rate']:.4f}",
                f"{s['std_completion_rate']:.4f}",
                f"{s['ci_completion_rate'][0]:.4f}",
                f"{s['ci_completion_rate'][1]:.4f}",
                f"{s['mean_deadlock_rate']:.4f}",
                f"{s['std_deadlock_rate']:.4f}",
                f"{s['ci_deadlock_rate'][0]:.4f}",
                f"{s['ci_deadlock_rate'][1]:.4f}",
                f"{s['mean_normalized_reward']:.4f}",
                f"{s['std_normalized_reward']:.4f}",
                s['perfect_episodes'],
                s['zero_deadlock_episodes'],
                s['n_episodes']
            ])
    
    print(f"CSV results saved to: {output_file}")


def save_json_results(summaries: Dict[str, Dict], output_file: str):
    """Save results to JSON file."""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy(i) for i in obj)
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_numpy(summaries), f, indent=2)
    
    print(f"JSON results saved to: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Deterministic Evaluation for Flatland RL Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_enhanced.py --agent DDDQNAgent --episodes 100
  python evaluate_enhanced.py --all-agents --episodes 100 --save-results
  python evaluate_enhanced.py --all-agents --episodes 100 --latex
        """
    )
    
    parser.add_argument('--agent', '-a', type=str, default='DDDQNAgent',
                        help='Agent class name')
    parser.add_argument('--all-agents', action='store_true',
                        help='Evaluate all DQN variants')
    parser.add_argument('--env', '-e', type=str, default='phase3_five_agents',
                        help='Environment name')
    parser.add_argument('--episodes', '-n', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Checkpoint path')
    parser.add_argument('--base-dir', type=str, default='',
                        help='Base directory')
    parser.add_argument('--save-results', action='store_true',
                        help='Save results to files')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX table')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Load configs
    env_configs = load_json_file(os.path.join(args.base_dir, DEFAULT_ENVIRONMENTS_FILE))
    setup_params = load_json_file(os.path.join(args.base_dir, DEFAULT_SETUP_FILE))
    
    if args.env not in env_configs:
        print(f"Error: Environment '{args.env}' not found")
        print(f"Available: {list(env_configs.keys())}")
        sys.exit(1)
    
    env_params = env_configs[args.env]
    
    # Determine agents to evaluate
    if args.all_agents:
        agents = AGENT_ORDER
    else:
        agents = [args.agent]
    
    # Run evaluations
    all_summaries = {}
    
    for agent_name in agents:
        if not args.quiet:
            print(f"\n{'#'*60}")
            print(f" Evaluating: {AGENT_DISPLAY_NAMES.get(agent_name, agent_name)}")
            print(f"{'#'*60}")
        
        try:
            evaluator = EnhancedEvaluator(
                agent_class_name=agent_name,
                env_params=env_params,
                setup_params=setup_params,
                checkpoint_path=args.checkpoint,
                base_dir=args.base_dir,
                use_action_masking=False,  # Match training settings
                seed=args.seed,
                verbose=not args.quiet
            )
            
            summary = evaluator.evaluate(n_episodes=args.episodes)
            evaluator.print_summary(summary)
            all_summaries[agent_name] = summary
            
        except Exception as e:
            print(f"Error evaluating {agent_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate outputs
    if len(all_summaries) > 1:
        generate_comparison_table(all_summaries)
    
    if args.latex:
        os.makedirs(args.output_dir, exist_ok=True)
        generate_latex_table(all_summaries, 
                           os.path.join(args.output_dir, "results_table.tex"))
    
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_json_results(all_summaries, 
                         os.path.join(args.output_dir, f"eval_results_{timestamp}.json"))
        generate_csv_results(all_summaries,
                            os.path.join(args.output_dir, f"eval_results_{timestamp}.csv"))
    
    return all_summaries


if __name__ == "__main__":
    main()