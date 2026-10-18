import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = 'CliffWalking-v1'
STEP_LIMIT = 200
RUNS = 5
EPISODE_COUNT = 500
BASELINE = dict(learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.01)
class QLearningAgent:

    def __init__(self, n_states=48, n_actions=4, learning_rate=0.1,
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.95,
                 epsilon_min=0.01):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

