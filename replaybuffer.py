import random
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        #original sampling: random.sample(self.memory, k=self.batch_size)
        #Redesigned the sampling to give higher weight to episodes with greater total rewards
        #Making all rewards positive to determine probabilities for sampling episode
        #reward_memory_list = [exp[2] for exp in self.memory]
        #min_reward_value = min(reward_memory_list)
        #max_reward_value = max(reward_memory_list)
        #reward_range = max_reward_value - min_reward_value
        #total_memory_reward = sum(exp[2]+np.abs(min_reward_value) for exp in self.memory)
        #Deriving the proportion of reward gained in each episode of total reward sum in memory
        #weights = [(exp[2]+np.abs(min_reward_value)) / total_memory_reward for exp in self.memory]
        #Build index list of episodes in memory using probabilities from weights
        #Using replacement to re-select more higher reward episodes
        #sample_idx = np.random.choice(len(self.memory), size=batch_size, p=weights, replace=True)
        #List comprehension pulling episodes from memory based on indicies above
        #sample = [self.memory[i] for i in sample_idx]
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
