import numpy as np
import matplotlib.pyplot as plt


class EpGrNonStatAgent:
    def __init__(self, n_arms, epsilon, alpha):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_values = np.zeros(n_arms)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.q_values)

    def update_q_value(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])


def band_nonstat(action, true_rewards):
    reward = np.random.randn() + true_rewards[action]
    return reward


def run_simulation(n_arms, n_steps, epsilon, alpha):
    true_rewards = np.zeros(n_arms)
    agent = EpGrNonStatAgent(n_arms, epsilon, alpha)

    rewards_history = []
    optimal_action_history = []

    for step in range(n_steps):
        action = agent.select_action()
        reward = band_nonstat(action, true_rewards)
        agent.update_q_value(action, reward)

        rewards_history.append(reward)
        optimal_action = np.argmax(true_rewards)
        optimal_action_history.append(action == optimal_action)

        true_rewards += np.random.normal(0, 0.01, n_arms)

    return rewards_history, optimal_action_history


n_arms = 10
n_steps = 10000
epsilon = 0.1
alpha = 0.1


rewards, optimal_actions = run_simulation(n_arms, n_steps, epsilon, alpha)


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(np.cumsum(rewards) / np.arange(1, len(rewards) + 1))
plt.title("Average Reward Over Time")
plt.xlabel("Steps")
plt.ylabel("Average Reward")

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(optimal_actions) / np.arange(1, len(optimal_actions) + 1))
plt.title("Optimal Action Percentage")
plt.xlabel("Steps")
plt.ylabel("Percentage")

plt.tight_layout()
plt.show()


print(f"Final Average Reward: {np.mean(rewards)}")
print(f"Optimal Action Percentage: {np.mean(optimal_actions) * 100:.2f}%")
