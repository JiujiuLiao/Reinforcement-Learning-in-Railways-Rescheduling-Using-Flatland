import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict


DEFAULT_LOG_DIR = "tmp/logs/wandb"
DEFAULT_OUTPUT_DIR = "diagnostic_plots"
WINDOW_SIZE = 100  # For moving averages


# =============================================================================
# DATA LOADING
# =============================================================================

def find_all_runs(log_root: str) -> list:
    """Find all training runs with episodes.csv files."""
    runs = []
    
    if not os.path.exists(log_root):
        print(f"WARNING: Log directory not found: {log_root}")
        return runs
    
    for root, dirs, files in os.walk(log_root):
        if "episodes.csv" in files:
            episodes_path = os.path.join(root, "episodes.csv")
            summary_path = os.path.join(root, "summary.json")
            run_name = os.path.basename(root)
            
            # Extract agent name and timestamp
            parts = run_name.split("-")
            if len(parts) >= 2:
                # Handle names like "d3qn-agent-20260130-072520"
                agent_name = "-".join(parts[:-2]) if len(parts) > 2 else parts[0]
                timestamp = "-".join(parts[-2:]) if len(parts) > 2 else parts[-1]
            else:
                agent_name = run_name
                timestamp = ""
            
            # Load summary for environment info
            env_name = "unknown"
            params = {}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    params = summary.get("params", {})
                    env_name = params.get("env", "unknown")
                except:
                    pass
            
            runs.append({
                "run_dir": root,
                "run_name": run_name,
                "episodes_path": episodes_path,
                "summary_path": summary_path,
                "agent": agent_name,
                "env": env_name,
                "timestamp": timestamp,
                "params": params
            })
    
    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x["timestamp"], reverse=True)
    return runs


def load_episodes_data(episodes_path: str) -> pd.DataFrame:
    """Load and clean episodes.csv data."""
    try:
        df = pd.read_csv(episodes_path)
        
        # Keep only episode rows
        if "type" in df.columns:
            df = df[df["type"] == "epsd"].copy()
        
        # Ensure idx is numeric
        df["idx"] = pd.to_numeric(df["idx"], errors="coerce")
        df = df.dropna(subset=["idx"])
        df = df.sort_values("idx")
        
        return df
    except Exception as e:
        print(f"Error loading {episodes_path}: {e}")
        return None


def load_all_runs_data(log_root: str) -> dict:
    """Load data from all runs."""
    runs = find_all_runs(log_root)
    all_data = {}
    
    for run in runs:
        df = load_episodes_data(run["episodes_path"])
        if df is not None and len(df) > 0:
            key = f"{run['agent']}_{run['env']}_{run['timestamp']}"
            all_data[key] = {
                "df": df,
                "info": run
            }
    
    return all_data


# =============================================================================
# DIAGNOSTIC ANALYSIS FUNCTIONS
# =============================================================================

def analyze_learning_progress(df: pd.DataFrame) -> dict:
    """Analyze if the agent is learning (metrics improving over time)."""
    results = {}
    
    # Split into early, middle, late phases
    n = len(df)
    early = df.head(n // 3)
    late = df.tail(n // 3)
    
    for metric in ["completions_avg", "deadlocks_avg", "scores_avg"]:
        if metric in df.columns:
            early_mean = early[metric].mean()
            late_mean = late[metric].mean()
            improvement = late_mean - early_mean
            
            results[metric] = {
                "early_mean": early_mean,
                "late_mean": late_mean,
                "improvement": improvement,
                "improving": improvement > 0 if "deadlock" not in metric else improvement < 0
            }
    
    return results


def analyze_exploration(df: pd.DataFrame) -> dict:
    """Analyze exploration rate (epsilon) over training."""
    results = {}
    
    if "exploration_prob_avg" in df.columns:
        eps = df["exploration_prob_avg"]
        results["start_epsilon"] = eps.iloc[0] if len(eps) > 0 else None
        results["end_epsilon"] = eps.iloc[-1] if len(eps) > 0 else None
        results["min_epsilon"] = eps.min()
        results["episodes_to_min"] = eps.idxmin() if not eps.isna().all() else None
    elif "exploration_prob_val" in df.columns:
        eps = df["exploration_prob_val"]
        results["start_epsilon"] = eps.iloc[0] if len(eps) > 0 else None
        results["end_epsilon"] = eps.iloc[-1] if len(eps) > 0 else None
        results["min_epsilon"] = eps.min()
    
    return results


def analyze_action_distribution(df: pd.DataFrame) -> dict:
    """Analyze action probabilities over training."""
    results = {}
    
    action_cols = [col for col in df.columns if col.startswith("act_")]
    
    if action_cols:
        # Get action names
        action_names = {
            "act_0": "DO_NOTHING",
            "act_1": "MOVE_LEFT",
            "act_2": "MOVE_FORWARD",
            "act_3": "MOVE_RIGHT",
            "act_4": "STOP_MOVING"
        }
        
        # Calculate mean probabilities
        for col in action_cols:
            if col in df.columns:
                action_name = action_names.get(col, col)
                
                # Early vs late comparison
                n = len(df)
                early = df.head(n // 3)[col].mean()
                late = df.tail(n // 3)[col].mean()
                overall = df[col].mean()
                
                results[action_name] = {
                    "early": early,
                    "late": late,
                    "overall": overall,
                    "change": late - early
                }
    
    return results


def analyze_deadlock_patterns(df: pd.DataFrame) -> dict:
    """Analyze when and how deadlocks occur."""
    results = {}
    
    if "deadlocks_avg" in df.columns:
        deadlocks = df["deadlocks_avg"]
        
        results["mean_deadlock_rate"] = deadlocks.mean()
        results["max_deadlock_rate"] = deadlocks.max()
        results["min_deadlock_rate"] = deadlocks.min()
        results["std_deadlock_rate"] = deadlocks.std()
        
        # Find episodes with zero deadlocks
        zero_deadlock_eps = (deadlocks < 0.01).sum()
        results["zero_deadlock_episodes"] = zero_deadlock_eps
        results["zero_deadlock_rate"] = zero_deadlock_eps / len(deadlocks)
        
        # Trend analysis
        n = len(df)
        if n > 100:
            early = deadlocks.head(n // 3).mean()
            late = deadlocks.tail(n // 3).mean()
            results["deadlock_trend"] = "improving" if late < early else "worsening"
            results["deadlock_change"] = late - early
    
    return results


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute correlations between metrics."""
    results = {}
    
    metric_pairs = [
        ("completions_avg", "deadlocks_avg"),
        ("completions_avg", "exploration_prob_avg"),
        ("deadlocks_avg", "exploration_prob_avg"),
    ]
    
    for m1, m2 in metric_pairs:
        if m1 in df.columns and m2 in df.columns:
            corr = df[m1].corr(df[m2])
            results[f"{m1}_vs_{m2}"] = corr
    
    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_learning_curves(df: pd.DataFrame, run_info: dict, output_dir: str):
    """Plot comprehensive learning curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Learning Curves: {run_info['agent']} on {run_info['env']}", fontsize=14)
    
    episodes = df["idx"]
    
    # 1. Completion Rate
    ax = axes[0, 0]
    if "completions_avg" in df.columns:
        ax.plot(episodes, df["completions_avg"], alpha=0.7, label="Completion Rate")
        ax.axhline(y=df["completions_avg"].mean(), color='r', linestyle='--', alpha=0.5, label=f"Mean: {df['completions_avg'].mean():.2%}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Completion Rate")
    ax.set_title("Completion Rate Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 2. Deadlock Rate
    ax = axes[0, 1]
    if "deadlocks_avg" in df.columns:
        ax.plot(episodes, df["deadlocks_avg"], alpha=0.7, color='red', label="Deadlock Rate")
        ax.axhline(y=df["deadlocks_avg"].mean(), color='darkred', linestyle='--', alpha=0.5, label=f"Mean: {df['deadlocks_avg'].mean():.2%}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Deadlock Rate")
    ax.set_title("Deadlock Rate Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 3. Score
    ax = axes[0, 2]
    if "scores_avg" in df.columns:
        ax.plot(episodes, df["scores_avg"], alpha=0.7, color='green', label="Score")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Average Score Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Exploration Rate
    ax = axes[1, 0]
    eps_col = "exploration_prob_avg" if "exploration_prob_avg" in df.columns else "exploration_prob_val"
    if eps_col in df.columns:
        ax.plot(episodes, df[eps_col], alpha=0.7, color='purple', label="Epsilon")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (Epsilon)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 5. Loss
    ax = axes[1, 1]
    if "loss_avg" in df.columns:
        loss = df["loss_avg"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(loss) > 0:
            ax.plot(df.loc[loss.index, "idx"], loss, alpha=0.7, color='orange', label="Loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Action Distribution
    ax = axes[1, 2]
    action_cols = [f"act_{i}_avg" if f"act_{i}_avg" in df.columns else f"act_{i}_val" 
                   for i in range(5)]
    action_cols = [c for c in action_cols if c in df.columns]
    
    if action_cols:
        action_names = ["DO_NOTHING", "LEFT", "FORWARD", "RIGHT", "STOP"]
        for i, col in enumerate(action_cols):
            if col in df.columns:
                ax.plot(episodes, df[col], alpha=0.7, label=action_names[i])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action Probability")
    ax.set_title("Action Distribution Over Training")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    filename = f"learning_curves_{run_info['agent']}_{run_info['env']}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")


def plot_completion_vs_deadlock(df: pd.DataFrame, run_info: dict, output_dir: str):
    """Plot completion rate vs deadlock rate scatter."""
    if "completions_avg" not in df.columns or "deadlocks_avg" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by episode number
    scatter = ax.scatter(
        df["deadlocks_avg"], 
        df["completions_avg"],
        c=df["idx"],
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    
    plt.colorbar(scatter, label='Episode')
    
    ax.set_xlabel("Deadlock Rate", fontsize=12)
    ax.set_ylabel("Completion Rate", fontsize=12)
    ax.set_title(f"Completion vs Deadlock: {run_info['agent']} on {run_info['env']}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add ideal zone annotation
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target: 80% completion')
    ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='Target: <20% deadlock')
    ax.legend()
    
    filename = f"completion_vs_deadlock_{run_info['agent']}_{run_info['env']}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")


def plot_action_evolution(df: pd.DataFrame, run_info: dict, output_dir: str):
    """Plot how action distribution evolves over training."""
    action_cols = []
    for i in range(5):
        if f"act_{i}_avg" in df.columns:
            action_cols.append(f"act_{i}_avg")
        elif f"act_{i}_val" in df.columns:
            action_cols.append(f"act_{i}_val")
    
    if not action_cols:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Action Distribution: {run_info['agent']} on {run_info['env']}", fontsize=14)
    
    action_names = ["DO_NOTHING", "LEFT", "FORWARD", "RIGHT", "STOP"]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # Stacked area chart
    ax = axes[0]
    
    # Prepare data for stacked plot
    data_matrix = []
    valid_names = []
    valid_colors = []
    
    for i, col in enumerate(action_cols):
        if col in df.columns:
            data_matrix.append(df[col].values)
            valid_names.append(action_names[i] if i < len(action_names) else col)
            valid_colors.append(colors[i] if i < len(colors) else 'gray')
    
    if data_matrix:
        ax.stackplot(df["idx"], data_matrix, labels=valid_names, colors=valid_colors, alpha=0.7)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Action Probability")
        ax.set_title("Action Distribution (Stacked)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Bar chart comparing early vs late
    ax = axes[1]
    
    n = len(df)
    early_means = []
    late_means = []
    
    for col in action_cols:
        if col in df.columns:
            early_means.append(df.head(n // 3)[col].mean())
            late_means.append(df.tail(n // 3)[col].mean())
    
    x = np.arange(len(valid_names))
    width = 0.35
    
    ax.bar(x - width/2, early_means, width, label='Early Training', color='lightblue')
    ax.bar(x + width/2, late_means, width, label='Late Training', color='darkblue')
    
    ax.set_ylabel("Action Probability")
    ax.set_title("Action Distribution: Early vs Late Training")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filename = f"action_evolution_{run_info['agent']}_{run_info['env']}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")


def plot_training_phases_comparison(all_data: dict, output_dir: str):
    """Compare metrics across different training phases/environments."""
    # Group by agent type
    agent_groups = defaultdict(list)
    
    for key, data in all_data.items():
        agent = data["info"]["agent"]
        agent_groups[agent].append(data)
    
    for agent, runs in agent_groups.items():
        if len(runs) < 2:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Training Phases Comparison: {agent}", fontsize=14)
        
        for run_data in runs:
            df = run_data["df"]
            env = run_data["info"]["env"]
            
            # Completion rate
            if "completions_avg" in df.columns:
                axes[0, 0].plot(df["idx"], df["completions_avg"], label=env, alpha=0.7)
            
            # Deadlock rate
            if "deadlocks_avg" in df.columns:
                axes[0, 1].plot(df["idx"], df["deadlocks_avg"], label=env, alpha=0.7)
            
            # Score
            if "scores_avg" in df.columns:
                axes[1, 0].plot(df["idx"], df["scores_avg"], label=env, alpha=0.7)
            
            # Exploration
            eps_col = "exploration_prob_avg" if "exploration_prob_avg" in df.columns else "exploration_prob_val"
            if eps_col in df.columns:
                axes[1, 1].plot(df["idx"], df[eps_col], label=env, alpha=0.7)
        
        axes[0, 0].set_title("Completion Rate")
        axes[0, 0].set_ylabel("Rate")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title("Deadlock Rate")
        axes[0, 1].set_ylabel("Rate")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title("Score")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title("Exploration Rate")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Epsilon")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"phase_comparison_{agent}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")


def plot_summary_dashboard(all_data: dict, output_dir: str):
    """Create a summary dashboard of all runs."""
    if not all_data:
        return
    
    # Collect summary statistics
    summaries = []
    
    for key, data in all_data.items():
        df = data["df"]
        info = data["info"]
        
        # Use last 100 episodes for summary
        recent = df.tail(100)
        
        summary = {
            "run": key,
            "agent": info["agent"],
            "env": info["env"],
            "episodes": len(df),
        }
        
        if "completions_avg" in recent.columns:
            summary["completion_rate"] = recent["completions_avg"].mean()
        if "deadlocks_avg" in recent.columns:
            summary["deadlock_rate"] = recent["deadlocks_avg"].mean()
        if "scores_avg" in recent.columns:
            summary["score"] = recent["scores_avg"].mean()
        
        summaries.append(summary)
    
    if not summaries:
        return
    
    # =========================================================================
    # SORT BY AGENT TYPE FIRST, THEN BY ENVIRONMENT
    # =========================================================================
    agent_order = ["dqn-agent", "double-dqn-agent", "dueling-dqn-agent", "d3qn-agent"]
    env_order = ["phase0_single", "phase1_two_agents", "phase2_three_agents", 
                 "phase3_five_agents", "phase4_seven_agents"]
    
    def sort_key(s):
        agent_idx = agent_order.index(s["agent"]) if s["agent"] in agent_order else 99
        env_idx = env_order.index(s["env"]) if s["env"] in env_order else 99
        return (agent_idx, env_idx)
    
    summaries = sorted(summaries, key=sort_key)
    # =========================================================================
    
    summary_df = pd.DataFrame(summaries)
    
    # =========================================================================
    # CREATE SEPARATE FIGURE FOR EACH METRIC (Much clearer!)
    # =========================================================================
    n_runs = len(summaries)
    
    # Assign colors by agent type
    agent_colors = {
        "dqn-agent": "#1f77b4",        # Blue
        "double-dqn-agent": "#ff7f0e", # Orange
        "dueling-dqn-agent": "#2ca02c", # Green
        "d3qn-agent": "#d62728",       # Red
    }
    bar_colors = [agent_colors.get(s["agent"], "gray") for s in summaries]
    
    # Short labels
    x_labels = [f"{s['env'].replace('phase', 'P').replace('_agents', '').replace('_single', '').replace('_two', '2').replace('_three', '3').replace('_five', '5').replace('_seven', '7')}" 
                for s in summaries]
    
    # =========================================================================
    # PLOT 1: Completion Rate
    # =========================================================================
    fig, ax = plt.subplots(figsize=(max(12, n_runs * 0.8), 7))
    
    x = np.arange(len(summaries))
    bars = ax.bar(x, summary_df["completion_rate"], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80% target')
    ax.set_ylabel("Completion Rate", fontsize=14)
    ax.set_title("Completion Rate by Agent & Environment (Last 100 Episodes)", fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Value labels
    for bar, val in zip(bars, summary_df["completion_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add agent group separators and labels
    current_agent = None
    group_starts = []
    for i, s in enumerate(summaries):
        if s["agent"] != current_agent:
            if current_agent is not None:
                ax.axvline(x=i - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            group_starts.append((i, s["agent"]))
            current_agent = s["agent"]
    
    # Add agent name labels at top
    for i, (start_idx, agent_name) in enumerate(group_starts):
        # Find end of this group
        if i + 1 < len(group_starts):
            end_idx = group_starts[i + 1][0] - 1
        else:
            end_idx = len(summaries) - 1
        
        mid_x = (start_idx + end_idx) / 2
        display_name = agent_name.replace("-agent", "").replace("-", " ").upper()
        ax.text(mid_x, 1.08, display_name, ha='center', va='bottom', fontsize=11, 
               fontweight='bold', color=agent_colors.get(agent_name, "black"))
    
    # Legend for target line
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_completion_rate.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_completion_rate.png")
    
    # =========================================================================
    # PLOT 2: Deadlock Rate
    # =========================================================================
    fig, ax = plt.subplots(figsize=(max(12, n_runs * 0.8), 7))
    
    bars = ax.bar(x, summary_df["deadlock_rate"], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0.2, color='green', linestyle='--', linewidth=2, label='20% target')
    ax.set_ylabel("Deadlock Rate", fontsize=14)
    ax.set_title("Deadlock Rate by Agent & Environment (Last 100 Episodes)", fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
    ax.set_ylim([0, max(summary_df["deadlock_rate"].max() * 1.2, 0.5)])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Value labels
    for bar, val in zip(bars, summary_df["deadlock_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add agent group separators
    current_agent = None
    for i, s in enumerate(summaries):
        if s["agent"] != current_agent:
            if current_agent is not None:
                ax.axvline(x=i - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            current_agent = s["agent"]
    
    # Agent labels at top
    for i, (start_idx, agent_name) in enumerate(group_starts):
        if i + 1 < len(group_starts):
            end_idx = group_starts[i + 1][0] - 1
        else:
            end_idx = len(summaries) - 1
        
        mid_x = (start_idx + end_idx) / 2
        display_name = agent_name.replace("-agent", "").replace("-", " ").upper()
        y_pos = max(summary_df["deadlock_rate"].max() * 1.15, 0.45)
        ax.text(mid_x, y_pos, display_name, ha='center', va='bottom', fontsize=11, 
               fontweight='bold', color=agent_colors.get(agent_name, "black"))
    
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_deadlock_rate.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_deadlock_rate.png")
    
    # =========================================================================
    # PLOT 3: Score
    # =========================================================================
    fig, ax = plt.subplots(figsize=(max(12, n_runs * 0.8), 7))
    
    bars = ax.bar(x, summary_df["score"], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel("Average Score", fontsize=14)
    ax.set_title("Score by Agent & Environment (Last 100 Episodes)", fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Value labels
    for bar, val in zip(bars, summary_df["score"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
               f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add agent group separators
    current_agent = None
    for i, s in enumerate(summaries):
        if s["agent"] != current_agent:
            if current_agent is not None:
                ax.axvline(x=i - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            current_agent = s["agent"]
    
    # Agent labels at top
    max_score = summary_df["score"].max()
    for i, (start_idx, agent_name) in enumerate(group_starts):
        if i + 1 < len(group_starts):
            end_idx = group_starts[i + 1][0] - 1
        else:
            end_idx = len(summaries) - 1
        
        mid_x = (start_idx + end_idx) / 2
        display_name = agent_name.replace("-agent", "").replace("-", " ").upper()
        ax.text(mid_x, max_score * 1.1, display_name, ha='center', va='bottom', fontsize=11, 
               fontweight='bold', color=agent_colors.get(agent_name, "black"))
    
    ax.set_ylim([0, max_score * 1.2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_score.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_score.png")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_diagnostic_report(all_data: dict, output_dir: str):
    """Generate a text report with diagnostic findings."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TRAINING DIAGNOSTICS REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for key, data in all_data.items():
        df = data["df"]
        info = data["info"]
        
        report_lines.append("-" * 80)
        report_lines.append(f"RUN: {key}")
        report_lines.append(f"Agent: {info['agent']}")
        report_lines.append(f"Environment: {info['env']}")
        report_lines.append(f"Episodes: {len(df)}")
        report_lines.append("-" * 80)
        
        # Learning Progress
        report_lines.append("\n[LEARNING PROGRESS]")
        progress = analyze_learning_progress(df)
        for metric, values in progress.items():
            status = "✓ IMPROVING" if values["improving"] else "✗ NOT IMPROVING"
            report_lines.append(f"  {metric}:")
            report_lines.append(f"    Early: {values['early_mean']:.4f}")
            report_lines.append(f"    Late:  {values['late_mean']:.4f}")
            report_lines.append(f"    Change: {values['improvement']:+.4f} {status}")
        
        # Exploration
        report_lines.append("\n[EXPLORATION]")
        exploration = analyze_exploration(df)
        if exploration:
            report_lines.append(f"  Start Epsilon: {exploration.get('start_epsilon', 'N/A')}")
            report_lines.append(f"  End Epsilon: {exploration.get('end_epsilon', 'N/A')}")
            report_lines.append(f"  Min Epsilon: {exploration.get('min_epsilon', 'N/A')}")
            
            if exploration.get('end_epsilon', 0) > 0.1:
                report_lines.append("  ⚠ WARNING: Epsilon still high - may need more training")
            elif exploration.get('end_epsilon', 1) < 0.01:
                report_lines.append("  ⚠ WARNING: Epsilon very low - limited exploration")
        
        # Action Distribution
        report_lines.append("\n[ACTION DISTRIBUTION]")
        actions = analyze_action_distribution(df)
        for action, values in actions.items():
            report_lines.append(f"  {action}:")
            report_lines.append(f"    Overall: {values['overall']:.2%}")
            report_lines.append(f"    Early→Late: {values['early']:.2%} → {values['late']:.2%} ({values['change']:+.2%})")
        
        # Deadlock Analysis
        report_lines.append("\n[DEADLOCK ANALYSIS]")
        deadlocks = analyze_deadlock_patterns(df)
        if deadlocks:
            report_lines.append(f"  Mean Deadlock Rate: {deadlocks.get('mean_deadlock_rate', 0):.2%}")
            report_lines.append(f"  Std Deadlock Rate: {deadlocks.get('std_deadlock_rate', 0):.2%}")
            report_lines.append(f"  Zero-Deadlock Episodes: {deadlocks.get('zero_deadlock_episodes', 0)} ({deadlocks.get('zero_deadlock_rate', 0):.1%})")
            
            if "deadlock_trend" in deadlocks:
                trend = deadlocks["deadlock_trend"]
                change = deadlocks["deadlock_change"]
                report_lines.append(f"  Trend: {trend.upper()} ({change:+.2%})")
                
                if deadlocks.get('mean_deadlock_rate', 0) > 0.5:
                    report_lines.append("  ⚠ CRITICAL: Deadlock rate > 50% - major coordination issue")
                elif deadlocks.get('mean_deadlock_rate', 0) > 0.3:
                    report_lines.append("  ⚠ WARNING: Deadlock rate > 30% - needs improvement")
        
        report_lines.append("")
    
    # Save report
    report_path = os.path.join(output_dir, "diagnostic_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nDiagnostic report saved: {report_path}")
    
    # Also print to console
    print('\n'.join(report_lines))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze training diagnostics")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR,
                        help="Directory containing training logs")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save diagnostic plots")
    parser.add_argument("--latest", type=int, default=None,
                        help="Only analyze the N most recent runs")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Only analyze first N episodes of each run")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("TRAINING DIAGNOSTICS ANALYSIS")
    print("=" * 60)
    print(f"Log directory: {args.log_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load all data
    print("Loading training data...")
    all_data = load_all_runs_data(args.log_dir)
    
    if args.max_episodes:
        print(f"Limiting analysis to first {args.max_episodes} episodes")
        for key in all_data:
            df = all_data[key]["df"]
            all_data[key]["df"] = df[df["idx"] <= args.max_episodes]

    if not all_data:
        print("No training data found!")
        return
    
    print(f"Found {len(all_data)} training runs")
    
    # Optionally limit to recent runs
    if args.latest:
        keys = list(all_data.keys())[:args.latest]
        all_data = {k: all_data[k] for k in keys}
        print(f"Analyzing {len(all_data)} most recent runs")
    
    print()
    
    # Generate plots for each run
    print("Generating diagnostic plots...")
    for key, data in all_data.items():
        print(f"\nProcessing: {key}")
        plot_learning_curves(data["df"], data["info"], args.output_dir)
        plot_completion_vs_deadlock(data["df"], data["info"], args.output_dir)
        plot_action_evolution(data["df"], data["info"], args.output_dir)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_training_phases_comparison(all_data, args.output_dir)
    plot_summary_dashboard(all_data, args.output_dir)
    
    # Generate text report
    print("\nGenerating diagnostic report...")
    generate_diagnostic_report(all_data, args.output_dir)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()