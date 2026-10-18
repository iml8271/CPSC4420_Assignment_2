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

def test_agent(agent, n_episodes=30):
    env = gym.make(ENV, max_episode_steps=STEP_LIMIT)
    test_rewards, steps, wins = [], [], 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward, count, done = 0, 0, False
        while not done:
            action = agent.get_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            count += 1
            done = terminated or truncated
        wins += terminated
        test_rewards.append(total_reward)
        steps.append(count)

    env.close()
    return np.mean(test_rewards), wins / n_episodes, np.mean(steps)


def moving_avg(x, window=20):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_results(episode_rewards, title, filename, window=20):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, label="Episode Reward")
    plt.plot(range(window - 1, len(episode_rewards)), moving_avg(episode_rewards, window),
             label=f"{window}-Episode Moving Avg")
    plt.ylim(-400, 0)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()


def run_experiment(name, values, label, filename):
    print(f"\n{label:>8} | episodes_til_-20 | avg_prev_100 | test_reward | success")
    plt.figure(figsize=(10, 5))
    best_value, best_speed = None, None

    for v in values:
        curves, speeds, results = [], [], []
        for seed in range(RUNS):
            agent, rewards = train_agent(seed=seed, **{name: v})
            curves.append(moving_avg(rewards))
            above = np.where(moving_avg(rewards) >= -20)[0]
            speeds.append(above[0] + 19 if len(above) else EPISODE_COUNT)
            results.append(test_agent(agent))

        avg_curve = np.mean(curves, axis=0)
        plt.plot(range(19, EPISODE_COUNT), avg_curve, label=f"{label} = {v}")

        speed = np.mean(speeds)
        last_100 = np.mean([c[-100:].mean() for c in curves])
        test_reward = np.mean([r[0] for r in results])
        success_rate = np.mean([r[1] for r in results])
        print(f"{v:>8} | {speed:20.0f} | {last_100:12.1f} | {test_reward:11.1f} | {success_rate:6.0%}")

        if best_speed is None or speed < best_speed:
            best_v, best_speed = v, speed

    plt.ylim(-400, 0)
    plt.xlabel("Episode")
    plt.ylabel(f"Reward (20-ep moving avg, {RUNS} runs)")
    plt.title(f"Effect of {label} on Q-Learning (CliffWalking)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    return best_v


if __name__ == "__main__":
    print("Baseline:", BASELINE)
    base_agent, base_rewards = train_agent()
    plot_results(base_rewards, "Q-Learning on Cliff Walking", "plot_C_baseline.png")
    base_test = test_agent(base_agent)

    best_lr = run_experiment("learning_rate", [0.1, 0.5, 0.9], "alpha", "plot_D_alpha.png")
    best_gamma = run_experiment("discount_factor", [0.5, 0.9, 0.99], "gamma", "plot_E_gamma.png")
    best_decay = run_experiment("epsilon_decay", [0.9, 0.95, 0.99], "eps_decay", "plot_F_epsilon_decay.png")

    best_cfg = dict(BASELINE, learning_rate=best_lr, discount_factor=best_gamma,
                    epsilon_decay=best_decay)
    print("\nbest cfg:", best_cfg)
    best_agent, best_rewards = train_agent(**best_cfg)
    plot_results(best_rewards, "Best cfg Q-Learning on CliffWalking", "plot_G_best.png")
    best_test = test_agent(best_agent)

    print(f"\n{'':10}{'avg reward':>12}{'success':>10}{'avg steps':>11}")
    for name, (r, s, n) in [("Base line", base_test), ("Best", best_test)]:
        print(f"{name:10}{r:12.1f}{s:10.0%}{n:11.1f}")