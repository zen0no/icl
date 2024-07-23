import numpy as np
import random


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def get_trajectory(self):
        return {
            "states": np.array(self.states, dtype=np.int32),
            "actions": np.array(self.actions, dtype=np.int32),
            "rewards": np.array(self.rewards, dtype=np.int32),
            "dones": np.array(self.dones, dtype=np.uint8),
        }


class QLearning:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_table = np.zeros((self.state_dim, self.action_dim))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            return np.argmax(self.q_table[state, :])

    def update(self, state, action, reward, next_state):
        self.q_table[state, action] += self.lr * (reward + self.gamma *
                                   (np.max(self.q_table[next_state, :]) - self.q_table[state, action]))
