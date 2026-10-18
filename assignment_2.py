import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

ENV = 'CliffWalking-v1'
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

    def update(self, state, action, reward, next_state, terminated):
        current_q = self.q_table[state, action]
        if terminated:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target_q - current_q)

        def decay_epsilon(self):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_agent(n_episodes=EPISODE_COUNT, seed=0, **settings):
        np.random.seed(seed)
        env = gym.make(ENV, max_episode_steps=STEP_LIMIT)
        agent = QLearningAgent(**{**BASELINE, **settings})
        episode_rewards = []

        for episode in range(n_episodes):
            state, _ = env.reset(seed=seed + episode)
            total_reward = 0
            done = False
            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                done = terminated or truncated
            agent.decay_epsilon()
            episode_rewards.append(total_reward)

        env.close()
        return agent, np.array(episode_rewards)

