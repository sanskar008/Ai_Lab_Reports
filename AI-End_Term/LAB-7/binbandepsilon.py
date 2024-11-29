import random
import numpy as np
import matplotlib.pyplot as plt


class BinaryBandit:
    def __init__(self, probabilities):
        self.num_arms = len(probabilities)
        self.probabilities = probabilities

    def avail_act(self):
        return list(range(self.num_arms))

    def pull_arm(self, action):
        value = random.random()
        return 1 if value < self.probabilities[action] else 0


def ep_greedy(bandit, epsilon, iterations):
    q_val = [0] * bandit.num_arms
    act_count = [0] * bandit.num_arms
    tot_rewards = []
    avg_rewards = [0]
    arm_selection = []

    for iteration in range(iterations):

        if random.random() > epsilon:
            chosen_action = q_val.index(max(q_val))
        else:
            chosen_action = random.choice(bandit.avail_act())

        reward = bandit.pull_arm(chosen_action)

        tot_rewards.append(reward)
        arm_selection.append(chosen_action)
        act_count[chosen_action] += 1

        q_val[chosen_action] += (reward - q_val[chosen_action]) / act_count[
            chosen_action
        ]

        avg_rewards.append(
            avg_rewards[-1] + (reward - avg_rewards[-1]) / (iteration + 1)
        )

    return {
        "q_values": q_val,
        "avg_rewards": avg_rewards[1:],
        "total_rewards": tot_rewards,
        "arm_selection": arm_selection,
    }


def plot_results(results_A, results_B):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(results_A["avg_rewards"], label="Bandit A")
    axes[0, 0].plot(results_B["avg_rewards"], label="Bandit B")
    axes[0, 0].set_title("Average Rewards vs Iterations")
    axes[0, 0].set_xlabel("Iterations")
    axes[0, 0].set_ylabel("Average Reward")
    axes[0, 0].legend()

    axes[0, 1].plot(results_A["total_rewards"], label="Bandit A", alpha=0.5)
    axes[0, 1].plot(results_B["total_rewards"], label="Bandit B", alpha=0.5)
    axes[0, 1].set_title("Rewards per Iteration")
    axes[0, 1].set_xlabel("Iterations")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].legend()

    axes[1, 0].bar(
        ["Arm 0", "Arm 1"],
        [results_A["arm_selection"].count(0), results_A["arm_selection"].count(1)],
        label="Bandit A",
    )
    axes[1, 0].set_title("Arm Selection Frequency (Bandit A)")
    axes[1, 0].set_ylabel("Number of Selections")

    axes[1, 1].bar(
        ["Arm 0", "Arm 1"],
        [results_B["arm_selection"].count(0), results_B["arm_selection"].count(1)],
        label="Bandit B",
    )
    axes[1, 1].set_title("Arm Selection Frequency (Bandit B)")
    axes[1, 1].set_ylabel("Number of Selections")

    plt.tight_layout()
    plt.show()

    print("Bandit A Final Q-values:", results_A["q_values"])
    print("Bandit B Final Q-values:", results_B["q_values"])
    print("\nBandit A Final Average Reward:", results_A["avg_rewards"][-1])
    print("Bandit B Final Average Reward:", results_B["avg_rewards"][-1])


random.seed(10)
np.random.seed(10)


bandit_A = BinaryBandit([0.1, 0.2])
bandit_B = BinaryBandit([0.8, 0.9])


results_A = ep_greedy(bandit_A, epsilon=0.2, iterations=2000)
results_B = ep_greedy(bandit_B, epsilon=0.2, iterations=2000)


plot_results(results_A, results_B)
