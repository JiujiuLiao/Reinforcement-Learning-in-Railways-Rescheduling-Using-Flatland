"""
DQN Training Demo for Flatland Railway Scheduling
This is a structural demo showing how to organize your DQN project.
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
from flatland.envs.observations import TreeObsForRailEnv
import matplotlib.pyplot as plt


# ============================================================================
# 1. DQN NETWORK ARCHITECTURE
# ============================================================================
class DQNNetwork(nn.Module):
    """Neural network for DQN agent"""
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# 2. REPLAY BUFFER
# ============================================================================
class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# 3. DQN AGENT
# ============================================================================
class DQNAgent:
    """DQN Agent with epsilon-greedy exploration"""
    def __init__(self, state_size, action_size, learning_rate=0.0001,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Main network and target network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.batch_size = 64
        self.update_target_every = 1000
        self.steps = 0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================================
# 4. ENVIRONMENT WRAPPER
# ============================================================================
class FlatlandWrapper:
    """Wrapper for Flatland environment to handle observations and actions"""
    def __init__(self, n_agents=3, width=30, height=30, max_depth=2):
        self.n_agents = n_agents
        
        # Create tree observation builder
        tree_obs_builder = TreeObsForRailEnv(max_depth=max_depth, predictor=None)
        
        # Create Flatland environment
        self.env = RailEnv(
            width=width,
            height=height,
            rail_generator=sparse_rail_generator(
                max_num_cities=3,
                grid_mode=False,
                max_rails_between_cities=2,
                max_rail_pairs_in_city=2
            ),
            line_generator=sparse_line_generator(),
            number_of_agents=n_agents,
            obs_builder_object=tree_obs_builder
        )
        
        # Action space: 5 actions per agent (DO_NOTHING, LEFT, FORWARD, RIGHT, STOP)
        self.action_size = 5
        self.state_size = 218  # TreeObs feature size
    
    def reset(self):
        """Reset environment and return initial state"""
        obs, info = self.env.reset()
        return self._process_observations(obs)
    
    def _process_observations(self, obs_dict):
        """
        Process raw observations into a single state vector.
        This is a simplified version - you'll need to adapt based on your approach.
        """
        # For demo: just take the first agent's observation
        # In practice, you might concatenate all agents or handle separately
        for agent_id, agent_obs in obs_dict.items():
            if agent_obs is not None:
                return np.array(agent_obs, dtype=np.float32)
        
        # Return zero observation if none available
        return np.zeros(self.state_size, dtype=np.float32)
    
    def step(self, action):
        """
        Take a step in the environment.
        This is simplified - you need to map single action to all agents.
        """
        # Create action dict for all agents
        # For demo: apply same action to all agents (you'll need better logic)
        action_dict = {i: action for i in range(self.n_agents)}
        
        obs, rewards, dones, info = self.env.step(action_dict)
        
        next_state = self._process_observations(obs)
        
        # Aggregate reward (sum of all agents)
        total_reward = sum(rewards.values())
        
        # Check if all agents are done
        all_done = dones['__all__']
        
        return next_state, total_reward, all_done, info
    
    def render(self):
        """Render the environment"""
        return self.env.show_render()


# ============================================================================
# 5. TRAINING LOOP
# ============================================================================
def train_dqn(n_episodes=1000, n_agents=3, curriculum_stages=None):
    """Main training loop"""
    
    # Initialize environment and agent
    env = FlatlandWrapper(n_agents=n_agents)
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01
    )
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    completion_rates = []
    losses = []
    
    print(f"Starting training with {n_agents} agents...")
    print(f"State size: {env.state_size}, Action size: {env.action_size}")
    print(f"Device: {agent.device}")
    
    # Training loop
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        episode_losses = []
        
        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Break if episode is too long (safety)
            if episode_length > 500:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        completion_rate = info.get('completion_rate', 0.0) if 'completion_rate' in info else 0.0
        completion_rates.append(completion_rate)
        
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_completion = np.mean(completion_rates[-10:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Completion: {avg_completion:.2%} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer Size: {len(agent.replay_buffer)}")
        
        # Save checkpoint
        if (episode + 1) % 100 == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
            }, f'/home/claude/checkpoint_ep{episode+1}.pth')
    
    return agent, episode_rewards, completion_rates, losses


# ============================================================================
# 6. EVALUATION FUNCTION
# ============================================================================
def evaluate_agent(agent, n_eval_episodes=10, n_agents=3):
    """Evaluate trained agent without exploration"""
    env = FlatlandWrapper(n_agents=n_agents)
    
    eval_rewards = []
    eval_completions = []
    
    print(f"\nEvaluating agent for {n_eval_episodes} episodes...")
    
    for episode in range(n_eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            action = agent.select_action(state, training=False)  # No exploration
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            steps += 1
        
        completion_rate = info.get('completion_rate', 0.0) if 'completion_rate' in info else 0.0
        eval_rewards.append(episode_reward)
        eval_completions.append(completion_rate)
        
        print(f"Eval Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Completion={completion_rate:.2%}")
    
    print(f"\nAverage Evaluation Reward: {np.mean(eval_rewards):.2f}")
    print(f"Average Completion Rate: {np.mean(eval_completions):.2%}")
    
    return eval_rewards, eval_completions


# ============================================================================
# 7. VISUALIZATION
# ============================================================================
def plot_training_metrics(episode_rewards, completion_rates, losses):
    """Plot training progress"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot rewards
    axes[0].plot(episode_rewards)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Rewards')
    axes[0].grid(True)
    
    # Plot completion rates
    axes[1].plot(completion_rates)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Completion Rate')
    axes[1].set_title('Training Completion Rates')
    axes[1].grid(True)
    
    # Plot losses
    if losses:
        axes[2].plot(losses)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss')
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/claude/training_metrics.png')
    print("Training metrics saved to training_metrics.png")


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Training configuration
    N_EPISODES = 500
    N_AGENTS = 3
    
    # Train the agent
    trained_agent, rewards, completions, losses = train_dqn(
        n_episodes=N_EPISODES,
        n_agents=N_AGENTS
    )
    
    # Evaluate the agent
    eval_rewards, eval_completions = evaluate_agent(
        trained_agent,
        n_eval_episodes=20,
        n_agents=N_AGENTS
    )
    
    # Visualize results
    plot_training_metrics(rewards, completions, losses)
    
    print("\nTraining complete!")
