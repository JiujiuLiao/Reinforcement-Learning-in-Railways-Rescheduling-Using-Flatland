from abc import ABC, abstractmethod
import random
from collections import deque

import numpy as np

from fltlnd.utils import SumTree


class Buffer(ABC):
    """
    Abstract base class for replay buffers.
    Defines the common interface used by different buffer implementations.
    """

    def __init__(self, buffer_size, batch_size):
        """
        Parameters
        ----------
        buffer_size : int
            Maximum number of transitions to store.
        batch_size : int
            Number of transitions to sample in each training batch.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        """Add a new transition to the buffer."""
        pass

    @abstractmethod
    def sample(self):
        """Sample a batch of transitions from the buffer."""
        pass

    @abstractmethod
    def update(self, idx, error):
        """
        Update internal state (e.g., priorities) using an error signal.
        Some buffers may ignore this (e.g., standard replay buffer).
        """
        pass

    @abstractmethod
    def add_agent_episode(self, agent, action, value, obs, reward, done, policy_logits):
        """
        Add a transition belonging to a specific agent's episode.
        Used by episodic / PPO-style buffers.
        """
        pass

    @abstractmethod
    def retrieve_agent_episodes(self, agent):
        """
        Retrieve all stored transitions for a given agent.
        """
        pass

    @abstractmethod
    def reset(self):
        """Clear all stored data in the buffer."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the current number of stored transitions."""
        pass


class ReplayBuffer(Buffer):
    """Standard replay buffer with a fixed capacity."""

    def __init__(self, buffer_size, batch_size):
        super().__init__(buffer_size, batch_size)
        # Capacity of the buffer (maximum number of entries)
        self._capacity = buffer_size
        # Internal storage of transitions
        self.memory = deque(maxlen=self._capacity)
        # Whether we store per-transition probabilities (for e.g. importance sampling)
        self.has_probability = False

    def add(self, state, action, reward, next_state, done, probability=None):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        state, action, reward, next_state, done :
            Standard experience tuple fields.
        probability : optional
            Optional probability associated with this transition.
            If provided at least once, `sample()` will also return it.
        """
        self.memory.append([state, action, reward, next_state, done, probability])
        if probability is not None:
            self.has_probability = True

    def get_last(self):
        """Return the most recently added transition."""
        return self.memory.__getitem__(self.memory.__len__() - 1)

    def sample(self):
        """
        Randomly sample a batch of transitions from memory.

        Returns
        -------
        If `has_probability` is False:
            (state, action, reward, next_state, done)
        If `has_probability` is True:
            (state, action, reward, next_state, done, probability)
        Each element is a NumPy array.
        """
        # Sample a mini-batch
        batch = random.sample(self.memory, self.batch_size)

        # Unzip batch into separate arrays
        state, action, reward, next_state, done, probability = [
            np.squeeze(i) for i in zip(*batch)
        ]

        if self.has_probability:
            return state, action, reward, next_state, done, probability
        else:
            return state, action, reward, next_state, done

    def update(self, error):
        """
        Standard replay buffer does not use priority,
        so this method is a no-op.
        """
        pass

    def add_agent_episode(self, agent, action, value, obs, reward, done, policy_logits):
        """
        Not implemented for this buffer type.

        This buffer is designed for flat transitions, not per-agent episodes.
        """
        raise NotImplementedError()

    def retrieve_agent_episodes(self, agent):
        """Not implemented for this buffer type."""
        raise NotImplementedError()

    def reset(self):
        """Clear all stored transitions."""
        self.memory.clear()
        self.has_probability = False

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedBuffer(Buffer):
    """
    Prioritized Experience Replay buffer implemented with a SumTree.
    """

    def __init__(self, buffer_size, batch_size):
        super().__init__(buffer_size, batch_size)

        self._internal_len = 0
        # Small constant to avoid zero priority
        self.eta = 0.01
        # Exponent for prioritization
        self.alpha = 0.6
        # Importance-sampling exponent
        self.beta = 0.4
        # Increment for beta per sample call
        self.beta_growth = 0.001

        self._batch_size = batch_size
        # SumTree capacity should match buffer_size
        self.tree = SumTree(buffer_size)

    def _get_priority(self, error):
        """Convert TD-error to a priority value."""
        return (error + self.eta) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.

        The initial priority is set to the current maximum priority to ensure
        new samples are likely to be picked at least once.
        """
        sample = [state, action, reward, next_state, done]

        # Find the maximum priority among existing entries
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # Avoid zero priority, otherwise this experience would never be sampled
        if max_priority == 0:
            max_priority = 1.0

        self._internal_len += 1
        self.tree.add(max_priority, sample)

    def sample(self):
        """
        Sample a batch of transitions according to their priorities.

        Returns
        -------
        state, action, reward, next_state, done : np.ndarray
        """
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        # Increase beta towards 1 over time (full importance sampling correction)
        self.beta = np.min([1.0, self.beta + self.beta_growth])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        # Store last sampled indices so `update` can adjust priorities
        self.sample_ids = idxs

        state, action, reward, next_state, done = [
            np.squeeze(i) for i in zip(*batch)
        ]
        return state, action, reward, next_state, done

    def update(self, error):
        """
        Update priorities for the most recently sampled transitions.

        Parameters
        ----------
        error : float or np.ndarray
            TD-error(s) corresponding to the last sampled batch.
        """
        p = self._get_priority(error)

        for idx in self.sample_ids:
            self.tree.update(idx, p)

    def add_agent_episode(self, agent, action, value, obs, reward, done, policy_logits):
        """Not implemented for this buffer type (transition-based)."""
        raise NotImplementedError()

    def retrieve_agent_episodes(self, agent):
        """Not implemented for this buffer type."""
        raise NotImplementedError()

    def reset(self):
        """Reset internal state (not fully implemented here)."""
        # Depending on SumTree implementation, you may want to re-create it:
        # self.tree = SumTree(self.buffer_size)
        self._internal_len = 0

    def __len__(self):
        """Return the number of stored transitions."""
        return self._internal_len


# class AgentEpisodeBuffer(Buffer):
#     """
#     Episode buffer tailored for PPO-style algorithms.

#     For each agent, it stores the whole episode as a sequence of:
#     [action, value, obs, reward, done, policy_logits]
#     """

#     def __init__(self, buffer_size, batch_size):
#         super().__init__(buffer_size, batch_size)
#         # key = agent_id, value = list of [action, value, obs, reward, done, policy_logits]
#         self._memory = {}

#     # For compatibility with the base interface (not used by PPO)
#     def add(self, state, action, reward, next_state, done):
#         """
#         Unused for PPO.

#         PPO relies on add_agent_episode instead of this method.
#         """
#         return

#     def sample(self):
#         """
#         Random sampling is not needed for PPO-style training.

#         Use retrieve_agent_episodes() instead.
#         """
#         raise NotImplementedError(
#             "AgentEpisodeBuffer is designed for PPO; "
#             "use retrieve_agent_episodes() instead of sample()."
#         )

#     def update(self, idx, error):
#         """
#         PPO does not require priority updates.

#         This method is a no-op for this buffer.
#         """
#         return

#     def add_agent_episode(self, agent, action, value, obs, reward, done, policy_logits):
#         """
#         Add a single transition for the given agent.

#         Parameters
#         ----------
#         agent : hashable
#             Agent identifier (e.g., agent_handle).
#         action, value, obs, reward, done, policy_logits :
#             Data associated with one step of the agent's episode.
#         """
#         agent_mem = self._memory.get(agent, [])
#         agent_mem.append([action, value, obs, reward, done, policy_logits])
#         self._memory[agent] = agent_mem

#     def retrieve_agent_episodes(self, agent):
#         """
#         Retrieve all stored transitions for a given agent.

#         Returns
#         -------
#         actions, values, obs, rewards, dones, logits
#             actions, values, obs, rewards, dones are NumPy arrays.
#             logits is returned as a list (PPOAgent can stack them as needed).
#         """
#         # If there is no stored episode for this agent, return empty arrays
#         if agent not in self._memory or len(self._memory[agent]) == 0:
#             return (
#                 np.array([]),
#                 np.array([]),
#                 np.array([]),
#                 np.array([]),
#                 np.array([]),
#                 [],
#             )

#         # Unzip from [[...], [...], ...] into 6 separate sequences
#         actions, values, obs, rewards, dones, logits = zip(*self._memory[agent])

#         actions = np.array(actions)
#         values = np.array(values)
#         obs = np.array(obs)
#         rewards = np.array(rewards)
#         dones = np.array(dones)
#         # logits are kept as a list; PPOAgent can stack/convert them as needed

#         return actions, values, obs, rewards, dones, logits

#     def reset(self):
#         """Clear all stored episodes for all agents."""
#         self._memory = {}

#     def __len__(self):
#         """Return the total number of transitions over all agents."""
#         return sum(len(v) for v in self._memory.values())
