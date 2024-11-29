import numpy as np


class GridWorldEnvironment:
    def __init__(self):
        # Environment configuration
        self.wall = [(1, 1)]
        self.possible_actions = ["L", "R", "U", "D"]
        self.terminate_states = [(1, 3), (2, 3)]

        # Action probabilities
        self.prob_actions = {"L": 0.25, "R": 0.25, "U": 0.25, "D": 0.25}

        # Environment action mappings
        self.environment_left = {"L": "D", "R": "U", "U": "L", "D": "R"}
        self.environment_right = {"L": "U", "R": "D", "U": "R", "D": "L"}

    def is_valid_state(self, i, j):
        """Check if a state is valid."""
        return (i, j) not in self.wall and 0 <= i < 3 and 0 <= j < 4

    def print_value_function(self, V):
        """Pretty print the value function."""
        for i in range(2, -1, -1):
            print("--- --- --- --- --- --- ---")
            for j in range(4):
                v = V[i][j]
                print(f" {v:.2f}|" if v >= 0 else f"{v:.2f}|", end="")
            print("")

    def get_next_state(self, action, i, j):
        """Compute next state based on action."""
        state_map = {"L": (i, j - 1), "R": (i, j + 1), "U": (i + 1, j), "D": (i - 1, j)}
        return state_map[action]

    def compute_value_function(self, i, j, reward, reward_matrix, V, discount_factor=1):
        """
        Compute value function for a state considering non-deterministic actions.

        Args:
            i, j: Current state coordinates
            reward: Base reward
            reward_matrix: Reward matrix
            V: Current value function
            discount_factor: Discount factor (gamma)
        """
        total_value = 0
        for action in self.possible_actions:
            # Desired action (80% probability)
            desired_state = self.get_next_state(action, i, j)
            desired_state_value = (
                (
                    reward_matrix[desired_state[0]][desired_state[1]]
                    + discount_factor * V[desired_state[0]][desired_state[1]]
                )
                if self.is_valid_state(*desired_state)
                else (reward_matrix[i][j] + discount_factor * V[i][j])
            )

            # Left environment action (10% probability)
            left_env_state = self.get_next_state(self.environment_left[action], i, j)
            left_env_value = (
                (
                    reward_matrix[left_env_state[0]][left_env_state[1]]
                    + discount_factor * V[left_env_state[0]][left_env_state[1]]
                )
                if self.is_valid_state(*left_env_state)
                else (reward_matrix[i][j] + discount_factor * V[i][j])
            )

            # Right environment action (10% probability)
            right_env_state = self.get_next_state(self.environment_right[action], i, j)
            right_env_value = (
                (
                    reward_matrix[right_env_state[0]][right_env_state[1]]
                    + discount_factor * V[right_env_state[0]][right_env_state[1]]
                )
                if self.is_valid_state(*right_env_state)
                else (reward_matrix[i][j] + discount_factor * V[i][j])
            )

            # Compute weighted value
            action_value = (
                0.8 * desired_state_value + 0.1 * left_env_value + 0.1 * right_env_value
            )

            total_value += action_value * self.prob_actions[action]

        return total_value

    def iterative_policy_evaluation(self, reward, theta=1e-7):
        """
        Perform iterative policy evaluation.

        Args:
            reward: Base reward for each state
            theta: Convergence threshold
        """
        # Initialize value function
        V = np.zeros((3, 4))

        # Create reward matrix
        reward_matrix = np.full((3, 4), reward)
        reward_matrix[2][3] = 1
        reward_matrix[1][3] = -1

        iterations = 0
        while True:
            delta = 0
            for i in range(3):
                for j in range(4):
                    state = (i, j)
                    if state in self.terminate_states or state in self.wall:
                        continue

                    v = V[i][j]
                    V[i][j] = self.compute_value_function(
                        i, j, reward, reward_matrix, V
                    )
                    delta = max(delta, abs(v - V[i][j]))

            iterations += 1
            if delta < theta:
                print(f"Total Iterations: {iterations}")
                break

        return V


def main():
    # Rewards to test
    rewards = [-0.04, -2, 0.1, 0.02, 1]

    # Create environment
    env = GridWorldEnvironment()

    # Evaluate for different rewards
    for reward in rewards:
        print(f"\nAt r(S): {reward}")
        value_function = env.iterative_policy_evaluation(reward)
        env.print_value_function(value_function)
        print("\n**************\n")


if __name__ == "__main__":
    main()
