import subprocess
import json
import sys
import os

CURRICULUM = [
    # Phase 0: Single agent pathfinding (baseline)
    ("phase0_single", 500, False),      # Fresh start, learn basic navigation
    
    # Phase 1: Two agents - learn basic collision avoidance
    ("phase1_two_agents", 1000, False), # Fresh start - don't transfer single-agent policy!
    
    # Phase 2: Three agents - more complex coordination
    ("phase2_three_agents", 1500, True), # Transfer from 2-agent policy
    
    # Phase 3: Five agents - scale up
    ("phase3_five_agents", 2000, True),  # Transfer from 3-agent policy
    
    # Phase 4: Add malfunctions
    ("phase4_five_malfunction", 1500, True),  # Same agents, add disruptions
    
    # Phase 5: Full scale
    ("phase5_ten_agents", 3000, True),   # Transfer to 10 agents
]

AGENTS = ["DQNAgent"]

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

    # Update setup.json
    with open("parameters/setup.json", "r") as f:
        cfg = json.load(f)
    cfg["sys"]["agent_class"] = agent
    cfg["trn"]["env"] = env

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
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"WARNING: Training failed for {env}")
        return False
    return True

def main():
    print("\n" + "="*60)
    print(" IMPROVED CURRICULUM TRAINING")
    print(" Key improvements:")
    print("   - Gentler transitions (1→2→3→5→10 agents)")
    print("   - Fresh start for multi-agent (no single-agent transfer)")
    print("   - More episodes per phase")
    print("   - Properly scaled rewards in handler.py")
    print("="*60)
    
    for agent in AGENTS:
        print(f"\n{'#'*60}")
        print(f" Starting curriculum for {agent}")
        print(f"{'#'*60}")
        
        for i, (env, episodes, transfer) in enumerate(CURRICULUM):
            print(f"\n>>> Phase {i}: {env}")
            success = run(agent, env, episodes, transfer_learning=transfer)
            
            if not success:
                print(f"Phase {i} failed, stopping curriculum")
                break
        
        print(f"\nCompleted curriculum for {agent}")

if __name__ == "__main__":
    main()