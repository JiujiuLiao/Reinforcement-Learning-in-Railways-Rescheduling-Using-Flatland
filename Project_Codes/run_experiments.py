import subprocess
import json
import sys
import os

# Phase-specific hyperparameters for curriculum learning
# Each phase has parameters tuned for its complexity level

PHASE_PARAMS = {
    "phase0_single": {
        "learning_rate": 3e-4,
        "batch_size": 64,
        "exp_start": 1.0,
        "exp_decay": 0.995,
        "exp_end": 0.05,
        "target_update": 500,
        "tau": 0.005,
        "memory_size": 50000,
        "buffer_min_size": 5000,
        "update_every": 4,
        "gamma": 0.95,
        "hidden_sizes": [128, 128]
    },
    "phase1_two_agents": {
        "learning_rate": 1e-4,
        "batch_size": 128,
        "exp_start": 1.0,
        "exp_decay": 0.997,
        "exp_end": 0.05,
        "target_update": 1000,
        "tau": 0.001,
        "memory_size": 100000,
        "buffer_min_size": 10000,
        "update_every": 4,
        "gamma": 0.97,
        "hidden_sizes": [256, 256]
    },
    "phase2_three_agents": {
        "learning_rate": 5e-5,
        "batch_size": 128,
        "exp_start": 1.0,
        "exp_decay": 0.998,
        "exp_end": 0.05,
        "target_update": 1500,
        "tau": 0.001,
        "memory_size": 100000,
        "buffer_min_size": 10000,
        "update_every": 4,
        "gamma": 0.99,
        "hidden_sizes": [256, 256]
    },
    "phase3_five_agents": {
        "learning_rate": 3e-5,
        "batch_size": 256,
        "exp_start": 0.5,
        "exp_decay": 0.999,
        "exp_end": 0.05,
        "target_update": 2000,
        "tau": 0.0005,
        "memory_size": 150000,
        "buffer_min_size": 15000,
        "update_every": 4,
        "gamma": 0.99,
        "hidden_sizes": [256, 256]
    },
    "phase4_seven_agents": {
        "learning_rate": 2e-5,
        "batch_size": 512,
        "exp_start": 0.4,
        "exp_decay": 0.9995,
        "exp_end": 0.05,
        "target_update": 2500,
        "tau": 0.0005,
        "memory_size": 200000,
        "buffer_min_size": 20000,
        "update_every": 4,
        "gamma": 0.99,
        "hidden_sizes": [512, 256]
    },
    "phase5_seven_malfunction": {
        "learning_rate": 1e-5,
        "batch_size": 512,
        "exp_start": 0.3,
        "exp_decay": 0.9995,
        "exp_end": 0.05,
        "target_update": 3000,
        "tau": 0.0003,
        "memory_size": 200000,
        "buffer_min_size": 20000,
        "update_every": 4,
        "gamma": 0.99,
        "hidden_sizes": [512, 256]
    },
    "phase6_ten_agents": {
        "learning_rate": 1e-5,
        "batch_size": 512,
        "exp_start": 0.3,
        "exp_decay": 0.9997,
        "exp_end": 0.03,
        "target_update": 3000,
        "tau": 0.0003,
        "memory_size": 300000,
        "buffer_min_size": 30000,
        "update_every": 4,
        "gamma": 0.99,
        "hidden_sizes": [512, 256]
    },
    "phase7_ten_malfunction": {
        "learning_rate": 5e-6,
        "batch_size": 512,
        "exp_start": 0.2,
        "exp_decay": 0.9998,
        "exp_end": 0.03,
        "target_update": 3000,
        "tau": 0.0002,
        "memory_size": 500000,
        "buffer_min_size": 50000,
        "update_every": 4,
        "gamma": 0.99,
        "hidden_sizes": [512, 256]
    }
}

# Curriculum definition: (env_name, episodes, transfer_learning)
CURRICULUM = [
    # Phase 0: Single agent pathfinding (baseline)
    ("phase0_single", 500, False),
    
    # Phase 1: Two agents - learn basic collision avoidance
    ("phase1_two_agents", 1500, False),  # Fresh start - don't transfer single-agent policy!
    
    # Phase 2: Three agents - more complex coordination
    ("phase2_three_agents", 2500, True),
    
    # Phase 3: Five agents - scale up
    ("phase3_five_agents", 3500, True),
    
    # Uncomment below phases as needed:
    # ("phase4_seven_agents", 5000, True),
    # ("phase5_seven_malfunction", 6000, True),
    # ("phase6_ten_agents", 8000, True),
    # ("phase7_ten_malfunction", 10000, True),
]

AGENTS = ["DQNAgent","DoubleDQNAgent", "DuelingDQNAgent", "DDDQNAgent"] #, "PPOAgent"  


def clear_checkpoints():
    """Remove old checkpoints to ensure fresh start"""
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pt') or f.endswith('.pth'):
                os.remove(os.path.join(checkpoint_dir, f))
                print(f"Removed old checkpoint: {f}")


def run(agent, env, episodes, transfer_learning=True):
    """
    Run training for one curriculum phase.
    
    Args:
        agent: Agent class name
        env: Environment name from environments.json
        episodes: Number of episodes to train
        transfer_learning: If True, load previous checkpoint. If False, start fresh.
    """
    print(f"\n{'='*60}")
    print(f"Running {agent} on {env} for {episodes} episodes")
    print(f"Transfer learning: {transfer_learning}")
    print(f"{'='*60}")

    # Load setup.json
    with open("parameters/setup.json", "r") as f:
        cfg = json.load(f)
    
    # Set agent and environment
    cfg["sys"]["agent_class"] = agent
    cfg["trn"]["env"] = env

    # Apply phase-specific parameters
    if env in PHASE_PARAMS:
        phase_params = PHASE_PARAMS[env]
        print(f"\nApplying phase-specific parameters for {env}:")
        for key, value in phase_params.items():
            cfg["trn"][key] = value
            print(f"  {key}: {value}")
    else:
        print(f"\nWARNING: No phase-specific parameters found for {env}, using defaults")

    # Save updated config
    with open("parameters/setup.json", "w") as f:
        json.dump(cfg, f, indent=4)

    # Choose training mode based on transfer_learning flag
    if transfer_learning:
        training_mode = "best"  # Load best checkpoint and continue training
    else:
        training_mode = "fresh"  # Start from scratch
    
    cmd = [
        sys.executable, "main.py",
        "--training", training_mode,
        "--episodes", str(episodes),
        "--verbose"
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"WARNING: Training failed for {env}")
        return False
    return True


def print_curriculum_summary():
    """Print a summary of the curriculum and parameters"""
    print("\n" + "="*80)
    print(" CURRICULUM TRAINING SUMMARY")
    print("="*80)
    
    for env, episodes, transfer in CURRICULUM:
        params = PHASE_PARAMS.get(env, {})
        print(f"\n{env}:")
        print(f"  Episodes: {episodes}, Transfer: {transfer}")
        if params:
            print(f"  LR: {params.get('learning_rate', 'default')}, "
                  f"Batch: {params.get('batch_size', 'default')}, "
                  f"Exp: {params.get('exp_start', 'default')}→{params.get('exp_end', 'default')}")
            print(f"  Memory: {params.get('memory_size', 'default')}, "
                  f"Target update: {params.get('target_update', 'default')}, "
                  f"Tau: {params.get('tau', 'default')}")
    print("\n" + "="*80)


def main():
    print("\n" + "="*80)
    print(" IMPROVED CURRICULUM TRAINING WITH PHASE-SPECIFIC HYPERPARAMETERS")
    print("="*80)
    print("\n Key improvements:")
    print("   - Gentler transitions (1→2→3→5→7→10 agents)")
    print("   - Fresh start for multi-agent (no single-agent transfer)")
    print("   - Phase-specific learning rates (decreasing with complexity)")
    print("   - Phase-specific exploration schedules")
    print("   - Phase-specific batch sizes (increasing with complexity)")
    print("   - Phase-specific memory buffer sizes")
    print("   - Phase-specific target network update frequencies")
    
    print_curriculum_summary()
    
    for agent in AGENTS:
        print(f"\n{'#'*80}")
        print(f" Starting curriculum for {agent}")
        print(f"{'#'*80}")
        
        for i, (env, episodes, transfer) in enumerate(CURRICULUM):
            print(f"\n>>> Phase {i}: {env}")
            print(f"    Episodes: {episodes}, Transfer: {transfer}")
            
            success = run(agent, env, episodes, transfer_learning=transfer)
            
            if not success:
                print(f"Phase {i} failed, stopping curriculum for {agent}")
                break
        
        print(f"\n{'='*60}")
        print(f" Completed curriculum for {agent}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()