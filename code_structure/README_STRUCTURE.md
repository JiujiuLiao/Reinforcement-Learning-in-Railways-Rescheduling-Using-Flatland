# DQN Flatland Training Structure - Guide

This demo shows you the complete structure of a DQN project for Flatland. Here's how everything fits together and what you need to customize.

## Code Structure Overview

```
dqn_flatland_demo.py
├── 1. DQNNetwork          - Neural network architecture
├── 2. ReplayBuffer        - Experience storage
├── 3. DQNAgent            - Main agent logic (action selection, training)
├── 4. FlatlandWrapper     - Environment interface
├── 5. train_dqn()         - Training loop
├── 6. evaluate_agent()    - Evaluation without exploration
├── 7. plot_training_metrics() - Visualization
└── 8. main execution      - Entry point
```

## What Each Component Does

### 1. **DQNNetwork** (Lines ~20-40)
- **Purpose**: The neural network that approximates Q-values
- **Customize**:
  - `hidden_sizes=[256, 256]` - Change network depth/width
  - Add dropout, batch normalization, different activations
  - Try CNN if you use image-based observations

### 2. **ReplayBuffer** (Lines ~45-65)
- **Purpose**: Stores past experiences for training
- **Customize**:
  - `capacity=10000` - Increase for more diverse experiences
  - Implement prioritized experience replay
  - Add multi-step returns

### 3. **DQNAgent** (Lines ~70-150)
- **Purpose**: Main agent that learns and selects actions
- **Key methods**:
  - `select_action()` - Epsilon-greedy exploration
  - `store_transition()` - Save experience
  - `train_step()` - Update network weights
  - `decay_epsilon()` - Reduce exploration over time
- **Customize**:
  - `epsilon_decay=0.99` - Your critical parameter!
  - `gamma=0.99` - Discount factor for future rewards
  - `learning_rate=0.0001` - Network learning speed
  - `batch_size=64` - Training batch size
  - Try Double DQN or Dueling DQN architectures

### 4. **FlatlandWrapper** (Lines ~155-220)
- **Purpose**: Bridge between Flatland and your DQN agent
- **CRITICAL SECTION TO CUSTOMIZE**:
  - `_process_observations()` - How you convert TreeObs to state
  - `step()` - How you map actions to multiple agents
  - Currently very simplified - this is where you implement:
    - Sequential action selection for multiple agents
    - Observation concatenation/selection strategy
    - Multi-agent coordination logic

### 5. **train_dqn()** (Lines ~225-320)
- **Purpose**: Main training loop
- **Flow**:
  1. Initialize environment and agent
  2. For each episode:
     - Reset environment
     - Loop until done:
       - Select action
       - Step environment
       - Store experience
       - Train network
     - Decay epsilon
     - Log metrics
  3. Save checkpoints
- **Customize**:
  - Add curriculum learning logic
  - Implement early stopping
  - Add validation episodes
  - Custom reward shaping

### 6. **evaluate_agent()** (Lines ~325-360)
- **Purpose**: Test agent without exploration
- Sets `training=False` so epsilon=0 (pure exploitation)
- Run on unseen scenarios to check generalization

### 7. **plot_training_metrics()** (Lines ~365-395)
- **Purpose**: Visualize learning progress
- Plots rewards, completion rates, losses
- Add more metrics as needed

### 8. **Main Execution** (Lines ~400-425)
- Entry point that ties everything together
- Configure hyperparameters here

---

## Key Areas You MUST Customize

### **PRIORITY 1: Multi-Agent Handling** (FlatlandWrapper)

The current demo is oversimplified - it treats multiple agents poorly. You need to decide:

**Option A: Single Centralized Agent**
```python
def _process_observations(self, obs_dict):
    # Concatenate all agents' observations
    all_obs = []
    for agent_id in range(self.n_agents):
        if obs_dict[agent_id] is not None:
            all_obs.extend(obs_dict[agent_id])
    return np.array(all_obs, dtype=np.float32)

def step(self, action):
    # Single action → map to all agents
    # This requires carefully designing the action space
    pass
```

**Option B: Sequential Action Selection**
```python
def step(self, actions_list):
    # actions_list = [action0, action1, action2, ...]
    action_dict = {i: actions_list[i] for i in range(self.n_agents)}
    obs, rewards, dones, info = self.env.step(action_dict)
    return obs, rewards, dones, info
```

**Option C: Independent Agents**
```python
# Create separate DQNAgent for each train
# Each agent only sees its own observation
# Coordinate through reward shaping
```

### **PRIORITY 2: State Representation**

TreeObs gives you a 218-dimensional vector. You need to:
```python
def _process_observations(self, obs_dict):
    # Current: just takes first agent
    # Better: normalize, handle None values, aggregate properly
    
    processed_obs = []
    for agent_id in range(self.n_agents):
        obs = obs_dict.get(agent_id, None)
        if obs is None:
            obs = np.zeros(218)  # Handle missing observations
        else:
            obs = np.array(obs, dtype=np.float32)
            # Normalize if needed
            obs = obs / np.linalg.norm(obs) if np.linalg.norm(obs) > 0 else obs
        processed_obs.extend(obs)
    
    return np.array(processed_obs, dtype=np.float32)
```

### **PRIORITY 3: Reward Shaping**

Currently just uses Flatland's default rewards. You should add:
```python
def _compute_custom_reward(self, obs, action, next_obs, done, info):
    reward = 0
    
    # Reward for progress toward goal
    if next_obs['distance_to_goal'] < obs['distance_to_goal']:
        reward += 1.0
    
    # Penalty for delays
    reward -= 0.1 * info['delays']
    
    # Big reward for completion
    if done and info['completion_rate'] == 1.0:
        reward += 10.0
    
    # Penalty for deadlock
    if done and info['completion_rate'] == 0.0:
        reward -= 5.0
    
    return reward
```

### **PRIORITY 4: Curriculum Learning**

Add progressive difficulty scaling:
```python
def train_dqn_curriculum(curriculum_stages):
    """
    curriculum_stages = [
        {'n_agents': 2, 'episodes': 200, 'width': 20, 'height': 20},
        {'n_agents': 5, 'episodes': 300, 'width': 30, 'height': 30},
        {'n_agents': 10, 'episodes': 500, 'width': 40, 'height': 40},
    ]
    """
    agent = None
    
    for stage in curriculum_stages:
        env = FlatlandWrapper(**stage)
        
        if agent is None:
            agent = DQNAgent(...)
        else:
            # Transfer learning: keep trained weights
            pass
        
        # Train on this stage
        train_on_stage(agent, env, stage['episodes'])
```

---

## How to Run and Modify

### **Step 1: Run the Demo As-Is**
```bash
python dqn_flatland_demo.py
```
This will train with 3 agents for 500 episodes (but won't work well due to simplifications).

### **Step 2: Start Customizing**

**First Customization** - Fix observation processing:
- Edit `_process_observations()` in `FlatlandWrapper`
- Handle None observations properly
- Decide on concatenation vs separate handling

**Second Customization** - Fix action mapping:
- Edit `step()` in `FlatlandWrapper`
- Implement proper multi-agent action handling

**Third Customization** - Add reward shaping:
- Create `_compute_custom_reward()` method
- Call it in `step()` to modify rewards

**Fourth Customization** - Tune hyperparameters:
- Start with `epsilon_decay=0.99` (you found this works)
- Adjust `learning_rate`, `batch_size`, `gamma`
- Try different network architectures

### **Step 3: Add Curriculum Learning**

Once basic training works with 2-3 agents, implement progressive scaling.

---

## Common Issues and Solutions

### Issue 1: Zero Rewards Every Episode
**Diagnosis**: Likely deadlock (trains not moving)
**Solutions**:
- Check action validity
- Verify observation processing
- Add deadlock detection
- Improve reward shaping

### Issue 2: Loss Not Decreasing
**Diagnosis**: Learning not happening
**Solutions**:
- Reduce learning rate
- Check gradient flow
- Verify Q-value targets
- Ensure sufficient exploration

### Issue 3: Training Unstable
**Diagnosis**: Updates too aggressive
**Solutions**:
- Use target network updates
- Gradient clipping
- Smaller learning rate
- Larger batch size

---

## Next Steps After This Demo

1. **Implement proper multi-agent handling** - This is critical!
2. **Add comprehensive logging** - Track Q-values, action distributions
3. **Implement visualization** - Render episodes, plot learning curves
4. **Add checkpointing** - Save/load models properly
5. **Create evaluation suite** - Test on diverse scenarios
6. **Implement curriculum learning** - Scale from 3 to 100 agents
7. **Compare with baselines** - MILP solvers, heuristics

---

## File Organization for Your Project

```
your_project/
├── config.py              # Hyperparameters, settings
├── models/
│   ├── dqn_network.py    # Neural network architectures
│   └── agent.py          # DQN agent class
├── environment/
│   ├── flatland_wrapper.py  # Environment interface
│   └── reward_shaping.py    # Custom reward functions
├── training/
│   ├── train.py          # Main training script
│   ├── curriculum.py     # Curriculum learning logic
│   └── utils.py          # Helper functions
├── evaluation/
│   ├── evaluate.py       # Evaluation scripts
│   └── visualize.py      # Plotting and rendering
├── experiments/
│   ├── exp1_basic.py     # Different experiment configs
│   └── exp2_scaled.py
└── checkpoints/          # Saved models
```

This structure keeps your code organized as it grows!
