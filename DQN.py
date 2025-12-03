"""
DQN with CUSTOM SIMPLE OBSERVATIONS
Since TreeObs is broken, we create our own simple observation from scratch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
import matplotlib.pyplot as plt


# ============================================================================
# SIMPLE DQN NETWORK
# ============================================================================
class SimpleDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(SimpleDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# REPLAY BUFFER
# ============================================================================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DQN AGENT
# ============================================================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = SimpleDQN(state_size, action_size).to(self.device)
        self.target_net = SimpleDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(50000)
        self.batch_size = 64
        self.update_target_every = 1000
        self.steps = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.loss_fn(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================================
# CUSTOM SIMPLE OBSERVATION WRAPPER
# ============================================================================
class CustomObsWrapper:
    """
    CUSTOM observations built from scratch using only:
    - Agent position
    - Agent target
    - Agent direction
    - Distance to target
    - Map boundaries
    
    NO TreeObs - we build it ourselves!
    """
    def __init__(self, n_agents=2, width=20, height=20):
        self.n_agents = n_agents
        self.width = width
        self.height = height
        
        # Create environment with a dummy observation builder
        # We'll use GlobalObs as placeholder but won't actually use it
        self.env = RailEnv(
            width=width,
            height=height,
            rail_generator=sparse_rail_generator(
                max_num_cities=3,
                grid_mode=True,  # Grid mode for simpler layout
                max_rails_between_cities=2,
                max_rail_pairs_in_city=2,
                seed=42
            ),
            line_generator=sparse_line_generator(),
            number_of_agents=n_agents,
            obs_builder_object=GlobalObsForRailEnv(),  # Dummy - we won't use it!
            random_seed=42
        )
        
        # Our custom observation size per agent:
        # 4 features: [normalized_x, normalized_y, normalized_target_x, normalized_target_y,
        #              dist_to_target, direction, can_move_forward, steps_remaining]
        self.state_size = 8  # Simple!
        self.action_size = 5
        
        self.prev_distances = {}
        self.max_steps = 200
        self.current_step = 0
        
        print(f"\n{'='*70}")
        print(f"CUSTOM OBSERVATION Environment:")
        print(f"  Agents: {n_agents}")
        print(f"  Map: {width}×{height}")
        print(f"  State size: {self.state_size} (custom simple observations)")
        print(f"  Features per agent:")
        print(f"    - Position (x, y) normalized")
        print(f"    - Target (x, y) normalized")
        print(f"    - Distance to target")
        print(f"    - Current direction (0-3)")
        print(f"    - Can move forward (0 or 1)")
        print(f"    - Steps remaining (normalized)")
        print(f"{'='*70}\n")
    
    def reset(self):
        obs, info = self.env.reset()
        self.current_step = 0
        self.prev_distances = {}
        
        for agent_id in range(self.n_agents):
            agent = self.env.agents[agent_id]
            if agent.position and agent.target:
                dist = self._manhattan_distance(agent.position, agent.target)
                self.prev_distances[agent_id] = dist
        
        return self._get_custom_observations(), info
    
    def _get_custom_observations(self):
        """Create simple custom observations"""
        obs_dict = {}
        
        for agent_id in range(self.n_agents):
            agent = self.env.agents[agent_id]
            
            if agent.position is None or agent.target is None:
                # Agent not active yet
                obs_dict[agent_id] = np.zeros(self.state_size, dtype=np.float32)
                continue
            
            # Extract simple features
            pos_x, pos_y = agent.position
            target_x, target_y = agent.target
            
            # Calculate distance
            distance = self._manhattan_distance(agent.position, agent.target)
            
            # Get direction (0=North, 1=East, 2=South, 3=West)
            direction = agent.direction
            
            # Check if can move forward (simplified - just check if not at edge)
            can_move = 1.0
            if direction == 0 and pos_y == 0:  # North at top edge
                can_move = 0.0
            elif direction == 1 and pos_x == self.width - 1:  # East at right edge
                can_move = 0.0
            elif direction == 2 and pos_y == self.height - 1:  # South at bottom edge
                can_move = 0.0
            elif direction == 3 and pos_x == 0:  # West at left edge
                can_move = 0.0
            
            # Build observation vector
            observation = np.array([
                pos_x / self.width,  # Normalized position x
                pos_y / self.height,  # Normalized position y
                target_x / self.width,  # Normalized target x
                target_y / self.height,  # Normalized target y
                distance / (self.width + self.height),  # Normalized distance
                direction / 3.0,  # Normalized direction
                can_move,  # Binary: can move forward
                (self.max_steps - self.current_step) / self.max_steps  # Time remaining
            ], dtype=np.float32)
            
            obs_dict[agent_id] = observation
        
        return obs_dict
    
    def step(self, actions_dict):
        self.current_step += 1
        obs, rewards, dones, info = self.env.step(actions_dict)
        next_obs = self._get_custom_observations()
        
        # Simple reward shaping
        shaped_rewards = {}
        for agent_id in range(self.n_agents):
            agent = self.env.agents[agent_id]
            reward = 0.0
            
            if agent.position and agent.target:
                current_dist = self._manhattan_distance(agent.position, agent.target)
                
                # Progress reward
                if agent_id in self.prev_distances:
                    prev_dist = self.prev_distances[agent_id]
                    progress = prev_dist - current_dist
                    
                    if progress > 0:
                        reward += 2.0  # Strong reward for moving closer
                    elif progress < 0:
                        reward -= 1.0  # Penalty for moving away
                
                self.prev_distances[agent_id] = current_dist
                
                # Completion bonus
                if agent.position == agent.target:
                    reward += 50.0
                    print(f"  🎯 Agent {agent_id} reached target!")
                else:
                    # Small time penalty
                    reward -= 0.1
            
            shaped_rewards[agent_id] = reward
        
        total_reward = sum(shaped_rewards.values())
        done = dones['__all__'] or self.current_step >= self.max_steps
        
        return next_obs, total_reward, done, info
    
    @staticmethod
    def _manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# ============================================================================
# TRAINING
# ============================================================================
def train_custom_obs(n_episodes=1000, n_agents=2):
    """Train with custom observations"""
    
    env = CustomObsWrapper(n_agents=n_agents, width=20, height=20)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    
    episode_rewards = []
    completion_rates = []
    losses = []
    
    print(f"Starting training with CUSTOM observations...")
    print(f"Device: {agent.device}\n")
    
    for episode in range(n_episodes):
        obs_dict, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        episode_losses = []
        
        while not done and steps < 200:
            # Each agent selects action
            actions = {}
            for agent_id in range(n_agents):
                if info['action_required'][agent_id]:
                    actions[agent_id] = agent.select_action(obs_dict[agent_id])
                else:
                    actions[agent_id] = 0
            
            next_obs, reward, done, info = env.step(actions)
            
            # Store experiences
            for agent_id in range(n_agents):
                if info['action_required'][agent_id]:
                    agent.store_transition(
                        obs_dict[agent_id],
                        actions[agent_id],
                        reward / n_agents,
                        next_obs[agent_id],
                        done
                    )
            
            # Train
            loss = agent.train_step()
            if loss:
                episode_losses.append(loss)
            
            obs_dict = next_obs
            episode_reward += reward
            steps += 1
        
        agent.decay_epsilon()
        
        # Calculate completion
        completed = sum(1 for a in env.env.agents if a.position == a.target)
        completion_rate = completed / n_agents
        
        episode_rewards.append(episode_reward)
        completion_rates.append(completion_rate)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_completion = np.mean(completion_rates[-10:])
            recent_perfect = sum(1 for c in completion_rates[-10:] if c == 1.0)
            
            print(f"Ep {episode+1:4d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Completion: {avg_completion:5.1%} ({recent_perfect}/10 perfect) | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Steps: {steps:3d}")
        
        # Check for improvement
        if (episode + 1) == 100:
            avg_completion_100 = np.mean(completion_rates[-100:])
            if avg_completion_100 > 0:
                print(f"\n✅ SUCCESS! Agents are learning! Completion: {avg_completion_100:.1%}\n")
            else:
                print(f"\n⚠️  Still 0% after 100 episodes. May need more training time.\n")
        
        # Save checkpoint
        if (episode + 1) % 500 == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'epsilon': agent.epsilon,
            }, f'checkpoint_custom_ep{episode+1}.pth')
    
    return agent, episode_rewards, completion_rates, losses


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(rewards, completions, losses):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    window = 50
    
    # Rewards
    axes[0].plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), ma, 'r-', linewidth=2, label='MA(50)')
    axes[0].axhline(0, color='green', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True)
    
    # Completions
    axes[1].plot(completions, alpha=0.3, label='Raw')
    if len(completions) >= window:
        ma = np.convolve(completions, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(completions)), ma, 'g-', linewidth=2, label='MA(50)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Completion Rate')
    axes[1].set_title('Completion Rates')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim([0, 1.1])
    
    # Loss
    if losses:
        axes[2].plot(losses)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss')
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_CUSTOM_OBS.png', dpi=150)
    print("\n✅ Plot saved to training_CUSTOM_OBS.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DQN with CUSTOM SIMPLE OBSERVATIONS")
    print("TreeObs is broken, so we build our own!")
    print("="*70)
    
    trained_agent, rewards, completions, losses = train_custom_obs(
        n_episodes=1000,
        n_agents=2
    )
    
    final_comp = np.mean(completions[-100:])
    final_reward = np.mean(rewards[-100:])
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final completion rate (last 100): {final_comp:.1%}")
    print(f"Final reward (last 100): {final_reward:.2f}")
    print(f"{'='*70}\n")
    
    plot_results(rewards, completions, losses)
    
    if final_comp > 0.5:
        print("✅ EXCELLENT! Agent learned successfully!")
        print("   Now you can scale up to more agents and larger maps.")
    elif final_comp > 0.2:
        print("✅ GOOD! Agent is learning, but needs more training.")
        print("   Try running for 2000-3000 episodes.")
    else:
        print("⚠️  Limited learning. The problem may be too hard for simple observations.")
        print("   Consider simplifying further or increasing training time.")