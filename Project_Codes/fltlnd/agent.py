from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ------------------------------
#  Base Agent
# ------------------------------


class Agent(ABC):
    """
    Base class for all agents.

    state_size: dimension of flattened observation
    action_size: number of discrete actions
    params: hyperparameters dictionary (same keys as before)
    memory_class: replay buffer class (the same class you used before)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        params: Dict[str, Any],
        memory_class,
        exploration: bool = True,
        train_best: bool = True,
        base_dir: str = "",
        checkpoint: str = None,
    ):
        self._state_size = state_size
        self._action_size = action_size
        self._params = params
        self._exploration = exploration
        self._base_dir = base_dir

        # device for PyTorch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._step_count = 0

        if checkpoint is not None:
            self.load(checkpoint)
        else:
            self.init_params()
            if train_best:
                self.load_best()
            else:
                self.create()

        # replay buffer
        self._memory = memory_class(self._memory_size, self._batch_size)

    # ---- interface that handler.py expects ----

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def init_params(self):
        pass

    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        pass

    @abstractmethod
    def step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        agent: int,
    ):
        pass

    @abstractmethod
    def save(self, filename: str, overwrite: bool = True):
        pass

    @abstractmethod
    def load(self, filename: str):
        pass

    def load_best(self):
        filename = os.path.join(self._base_dir, "checkpoints", str(self))
        if os.path.exists(filename):
            self.load(filename)
        else:
            self.create()

    def save_best(self):
        self.save(os.path.join(self._base_dir, "checkpoints", str(self)))

    @abstractmethod
    def step_start(self):
        pass

    @abstractmethod
    def episode_start(self):
        pass

    @abstractmethod
    def episode_end(self, agents: List[int]):
        pass

    @abstractmethod
    def __str__(self):
        pass


# ------------------------------
#  Random Agent
# ------------------------------


class RandomAgent(Agent):
    def act(self, obs):
        self.stats["eps_counter"] += 1
        return int(np.random.choice(np.arange(self._action_size)))

    def step(self, obs, action, reward, next_obs, done, agent):
        pass

    def save(self, filename, overwrite: bool = True):
        pass

    def load(self, filename):
        pass

    def load_best(self):
        self.create()

    def step_start(self):
        pass

    def episode_start(self):
        pass

    def episode_end(self, agents):
        pass

    def create(self):
        self.init_params()

    def init_params(self):
        self.stats = {"eps_val": 1.0, "eps_counter": 0, "loss": 0.0}
        self._memory_size = self._params["memory_size"]
        self._batch_size = self._params["batch_size"]
        self._buffer_min_size = self._params["buffer_min_size"]  # ← FIX THIS, this is new added

    def __str__(self):
        return "random-agent"


class FIFOAgent(Agent):
    """
    A very simple deterministic baseline:
    - If the environment says action_required = True ⇒ typically move forward (MOVE_FORWARD = 2)
    - Otherwise ⇒ DO_NOTHING = 0

    This is not a full FIFO policy on the whole network,
    but a simple rule-based policy used as a baseline alongside the Random agent.
    """

    def act(self, obs):
        # This agent does not use the observation in a smart way; it always moves forward.
        # (If the action is invalid, Flatland handles it.)
        self.stats["eps_counter"] += 0  # kept for interface compatibility
        return 2  # MOVE_FORWARD

    def step(self, obs, action, reward, next_obs, done, agent):
        # No learning ⇒ replay buffer is not used
        pass

    def save(self, filename, overwrite: bool = True):
        # No model ⇒ nothing to save
        pass

    def load(self, filename):
        # No model ⇒ nothing to load
        pass

    def load_best(self):
        self.create()

    def step_start(self):
        pass

    def episode_start(self):
        pass

    def episode_end(self, agents):
        pass

    def create(self):
        self.init_params()

    def init_params(self):
        self.stats = {"eps_val": 0.0, "eps_counter": 0, "loss": 0.0}
        self._memory_size = 1
        self._batch_size = 1

    def __str__(self):
        return "fifo-agent"


# ------------------------------
#  NN base agent
# ------------------------------
class NNAgent(Agent):
    def init_params(self):
        self._memory_size = self._params["memory_size"]
        self._batch_size = self._params["batch_size"]
        self._update_every = self._params["update_every"]
        self._learning_rate = self._params["learning_rate"]
        self._gamma = self._params["gamma"]
        self._buffer_min_size = self._params["batch_size"]
        self._hidden_sizes = self._params["hidden_sizes"]


# ------------------------------
#  Simple DQN network (PyTorch)
# ------------------------------


class DQNNet(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        last = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------
#  DQN Agent (SB3-like behaviour)
# ------------------------------


class DQNAgent(NNAgent):
    def act(self, obs: np.ndarray) -> int:
        """
        Epsilon-greedy action selection on a PyTorch network.
        """
        if (
            self.stats["eps_val"] > np.random.rand()
            and self._exploration
            and not self.noisy_net
        ):
            action = int(np.random.choice(self._action_size))
            self.stats["eps_counter"] += 1
            return action

        state_tensor = (
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self._device)
        )
        with torch.no_grad():
            q_values = self._model(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        return action

    def step(self, obs, action, reward, next_obs, done, agent):
        self._step_count += 1

        # Store transition in the replay buffer
        self._memory.add(obs, action, reward, next_obs, done)

        # Train every N steps
        if (
            self._step_count % self._update_every == 0
            and len(self._memory) > self._buffer_min_size
            and len(self._memory) > self._batch_size
        ):
            self.train()

    def train(self):
        (
            state_sample,
            action_sample,
            rewards_sample,
            state_next_sample,
            done_sample,
        ) = self._memory.sample()

        # numpy → torch
        states = torch.as_tensor(state_sample, dtype=torch.float32).to(self._device)
        actions = torch.as_tensor(action_sample, dtype=torch.long).to(self._device)
        rewards = torch.as_tensor(rewards_sample, dtype=torch.float32).to(self._device)
        next_states = torch.as_tensor(state_next_sample, dtype=torch.float32).to(
            self._device
        )
        dones = torch.as_tensor(done_sample, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            future_rewards = self._get_future_rewards(next_states)
            max_future_q = future_rewards.max(dim=1)[0]
            updated_q_values = rewards + self._gamma * max_future_q
            # Zero out future contribution for terminal states
            updated_q_values = updated_q_values * (1 - dones) - dones

        # Q(s,a)
        q_values = self._model(states)
        q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self._loss_fn(q_action, updated_q_values)
        self.stats["loss"] = float(loss.item())

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()

        # If replay buffer is prioritized
        if hasattr(self._memory, "update"):
            self._memory.update(loss.detach().cpu().numpy())

    def save(self, filename, overwrite: bool = True):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self._model.state_dict(), filename + ".pt")

    def load(self, filename):
        self.init_params()
        self._model = self.build_network().to(self._device)
        state_dict = torch.load(filename + ".pt", map_location=self._device)
        self._model.load_state_dict(state_dict)

    def create(self):
        self._model = self.build_network().to(self._device)

    def init_params(self):
        super().init_params()

        self.stats = {
            "eps_val": float(self._params["exp_start"]),
            "eps_counter": 0,
            "loss": None,
        }

        self.noisy_net = bool(self._params.get("noisy_net", False))
        self._eps_end = self._params["exp_end"]
        self._eps_decay = self._params["exp_decay"]
        self._tau = self._params["tau"]

        self._loss_fn = nn.SmoothL1Loss()  # Huber loss

        # The actual model instance is created in create(), but we need a
        # temporary network here to initialize the optimizer.
        self._model = self.build_network().to(self._device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._learning_rate
        )

    def build_network(self) -> nn.Module:
        return DQNNet(self._state_size, self._action_size, self._hidden_sizes)

    def step_start(self):
        pass

    def episode_start(self):
        self.stats["eps_counter"] = 0

    def episode_end(self, agents):
        # Epsilon decay
        self.stats["eps_val"] = max(
            self._eps_end, self._eps_decay * self.stats["eps_val"]
        )

    def _get_future_rewards(self, next_states: torch.Tensor) -> torch.Tensor:
        return self._model(next_states)

    def __str__(self):
        return "dqn-agent"

# ------------------------------
#  Double DQN
# ------------------------------


class DoubleDQNAgent(DQNAgent):
    def step(self, obs, action, reward, next_obs, done, agent):
        super().step(obs, action, reward, next_obs, done, agent)
        if self._step_count % self._target_update == 0 or self._soft_update:
            # Update target network
            with torch.no_grad():
                for target_param, param in zip(
                    self._model_target.parameters(), self._model.parameters()
                ):
                    if self._soft_update:
                        target_param.data.copy_(
                            self._tau * param.data + (1 - self._tau) * target_param.data
                        )
                    else:
                        target_param.data.copy_(param.data)

    def load(self, filename):
        super().load(filename)
        self._create_target()

    def create(self):
        super().create()
        self._create_target()

    def init_params(self):
        super().init_params()
        self._tau = self._params["tau"]
        self._target_update = self._params["target_update"]
        self._soft_update = self._params["soft_update"]

    def _create_target(self):
        self._model_target = self.build_network().to(self._device)
        self._model_target.load_state_dict(self._model.state_dict())

    def _get_future_rewards(self, next_states: torch.Tensor) -> torch.Tensor:
        # Double DQN with a target network
        return self._model_target(next_states)

    def __str__(self):
        return "double-dqn-agent"


# ------------------------------
#  Dueling DQN (PyTorch)
# ------------------------------


class DuelingDQNNet(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = self.value(x)  # [B, 1]
        a = self.advantage(x)  # [B, A]
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)
        return q


class DuelingDQNAgent(DQNAgent):
    def build_network(self) -> nn.Module:
        return DuelingDQNNet(self._state_size, self._action_size)

    def __str__(self):
        return "dueling-dqn-agent"


class DDDQNAgent(DuelingDQNAgent, DoubleDQNAgent):
    def __str__(self):
        return "dddqn-agent"


# ------------------------------
#  PPO (PyTorch)
# ------------------------------


class PPOModel(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.policy_head = nn.Linear(64, action_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        logits = self.policy_head(x)
        return value, logits

    def action_value(self, state: np.ndarray, device: torch.device):
        state_t = torch.as_tensor(state, dtype=torch.float32).to(device)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            value, logits = self(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.cpu().numpy(), value.cpu().numpy()


class PPOAgent(NNAgent):
    def __init__(
        self,
        state_size,
        action_size,
        params,
        memory_class,
        exploration,
        train_best,
        base_dir,
        checkpoint,
    ):
        self._last_value = None
        super().__init__(
            state_size,
            action_size,
            params,
            memory_class,
            exploration=exploration,
            train_best=train_best,
            base_dir=base_dir,
            checkpoint=checkpoint,
        )

    def act(self, obs: np.ndarray) -> int:
        actions, value = self._model.action_value(obs.reshape(1, -1), self._device)
        self._last_value = value
        return int(actions[0])

    def step(self, obs, action, reward, next_obs, done, agent):
        self._step_count += 1

        obs_t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32).to(
            self._device
        )
        with torch.no_grad():
            _, policy_logits = self._model(obs_t)

        if self._last_value is None:
            _, self._last_value = self._model.action_value(
                obs.reshape(1, -1), self._device
            )

        # memory_class must implement the same interface as before
        self._memory.add_agent_episode(
            agent,
            action,
            self._last_value[0],
            obs,
            reward,
            done,
            policy_logits.cpu().squeeze(0),
        )
        self._last_value = None

    def train(self, agents):
        # Decrease entropy weight over time
        self._entropy_weight = self._entropy_decay * self._entropy_weight

        for agent in agents:
            (
                actions,
                values,
                states,
                rewards,
                dones,
                probs,
            ) = self._memory.retrieve_agent_episodes(agent)

            states_t = torch.as_tensor(np.stack(states), dtype=torch.float32).to(
                self._device
            )
            values_t = torch.as_tensor(values, dtype=torch.float32).to(self._device)
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32).to(self._device)
            dones_t = torch.as_tensor(dones, dtype=torch.float32).to(self._device)

            # Bootstrap with value of the last state
            with torch.no_grad():
                _, next_value = self._model(
                    torch.as_tensor(states[-1].reshape(1, -1), dtype=torch.float32).to(
                        self._device
                    )
                )
            discounted_rewards, advantages = self._get_advantages(
                rewards_t, dones_t, values_t, next_value[0]
            )

            actions_t = torch.as_tensor(actions, dtype=torch.long).to(self._device)
            old_logits = torch.stack(probs).to(self._device)
            old_probs = F.softmax(old_logits, dim=-1)
            action_inds = torch.stack(
                [torch.arange(actions_t.shape[0]).to(self._device), actions_t], dim=1
            )
            old_probs_act = old_probs.gather(1, action_inds[:, 1].unsqueeze(1)).squeeze(
                1
            )

            # Gradient step
            self._optimizer.zero_grad()
            values_pred, policy_logits = self._model(states_t)
            act_loss = self._actor_loss(
                advantages, old_probs_act, action_inds, policy_logits
            )
            ent_loss = self._entropy_loss(policy_logits, self._entropy_weight)
            c_loss = self._critic_loss(discounted_rewards, values_pred.squeeze(1))
            tot_loss = act_loss + ent_loss + c_loss
            self.stats["loss"] = float(tot_loss.item())
            tot_loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=0.5)
            self._optimizer.step()

        self._memory.reset()

    def save(self, filename, overwrite: bool = True):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self._model.state_dict(), filename + ".pt")

    def load(self, filename):
        self.init_params()
        self._model = self._build_network().to(self._device)
        state_dict = torch.load(filename + ".pt", map_location=self._device)
        self._model.load_state_dict(state_dict)

    def create(self):
        self._model = self._build_network().to(self._device)

    def init_params(self):
        super().init_params()
        self.stats = {"loss": None, "eps_val": 0.0, "eps_counter": 0}

        self._learning_rate = self._params["learning_rate"]
        self._surrogate_eps_clip = self._params["surrogate_eps_clip"]
        self._loss_weight = self._params["loss_weight"]
        self._entropy_weight = self._params["entropy_weight"]
        self._entropy_decay = 0.998

        self._model = self._build_network().to(self._device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._learning_rate
        )

    def step_start(self):
        pass

    def episode_start(self):
        pass

    def episode_end(self, agents):
        self.train(agents)

    def _get_advantages(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
    ):
        rewards = rewards.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()
        values = values.detach().cpu().numpy()
        next_value = next_value.detach().cpu().numpy()

        # Generalized advantage estimation (simple discounted returns + baseline)
        discounted_rewards = np.array(list(rewards) + [next_value[0]])
        for t in reversed(range(len(rewards))):
            discounted_rewards[t] = (
                rewards[t] + 0.99 * discounted_rewards[t + 1] * (1 - dones[t])
            )
        discounted_rewards = discounted_rewards[:-1]
        advantages = discounted_rewards - values

        # Normalize advantages
        advantages -= advantages.mean()
        advantages /= (advantages.std() + 1e-10)

        # Normalize discounted rewards
        discounted_rewards -= discounted_rewards.mean()
        discounted_rewards /= (discounted_rewards.std() + 1e-8)

        dr_t = torch.as_tensor(discounted_rewards, dtype=torch.float32).to(self._device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32).to(self._device)
        return dr_t, adv_t

    def _build_network(self):
        return PPOModel(self._state_size, self._action_size)

    def _actor_loss(
        self,
        advantages: torch.Tensor,
        old_probs_act: torch.Tensor,
        action_inds: torch.Tensor,
        policy_logits: torch.Tensor,
    ):
        probs = F.softmax(policy_logits, dim=-1)
        new_probs_act = probs.gather(1, action_inds[:, 1].unsqueeze(1)).squeeze(1)

        ratio = new_probs_act / (old_probs_act + 1e-8)
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio,
            1.0 - self._surrogate_eps_clip,
            1.0 + self._surrogate_eps_clip,
        ) * advantages
        policy_loss = -torch.mean(torch.min(unclipped, clipped))
        return policy_loss

    def _critic_loss(
        self, discounted_rewards: torch.Tensor, value_est: torch.Tensor
    ) -> torch.Tensor:
        return (
            F.mse_loss(discounted_rewards, value_est) * float(self._loss_weight)
        ).to(torch.float32)

    def _entropy_loss(self, policy_logits: torch.Tensor, ent_discount_val: float):
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        return -entropy * ent_discount_val

    def __str__(self):
        return "ppo"