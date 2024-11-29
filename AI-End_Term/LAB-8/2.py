import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson


class JackCarRentalParameters:
    

    MAX_CARS = 20
    DISCOUNT_FACTOR = 0.9
    RENTAL_CREDIT = 10
    MOVING_COST = -2
    CONVERGENCE_THRESHOLD = 0.01
    MAX_TRANSFER = 5


class PoissonDistribution:
   
    def __init__(self, λ, epsilon=0.01):
        self.λ = λ
        self.probabilities = self._compute_probabilities(epsilon)

    def _compute_probabilities(self, epsilon):
        probabilities = {}
        cumulative_prob = 0
        k = 0

        while True:
            prob = poisson.pmf(k, self.λ)
            if prob <= epsilon:
                break

            probabilities[k] = prob
            cumulative_prob += prob
            k += 1

        correction = (1 - cumulative_prob) / len(probabilities)
        return {k: p + correction for k, p in probabilities.items()}

    def probability(self, n):
      return self.probabilities.get(n, 0)


class Location:
   
    def __init__(self, request_rate, return_rate):
        self.request_poisson = PoissonDistribution(request_rate)
        self.return_poisson = PoissonDistribution(return_rate)


class JackCarRentalEnvironment:
   
    def __init__(self, location_a, location_b):
        self.location_a = location_a
        self.location_b = location_b
        self.value_function = np.zeros(
            (JackCarRentalParameters.MAX_CARS + 1, JackCarRentalParameters.MAX_CARS + 1)
        )
        self.policy = np.zeros_like(self.value_function, dtype=int)

    def compute_expected_reward(self, state, action):
      
        new_state = [
            max(min(state[0] - action, JackCarRentalParameters.MAX_CARS), 0),
            max(min(state[1] + action, JackCarRentalParameters.MAX_CARS), 0),
        ]

        total_reward = JackCarRentalParameters.MOVING_COST * abs(action)

        for req_a in self.location_a.request_poisson.probabilities:
            for req_b in self.location_b.request_poisson.probabilities:
                for ret_a in self.location_a.return_poisson.probabilities:
                    for ret_b in self.location_b.return_poisson.probabilities:

                        scenario_prob = (
                            self.location_a.request_poisson.probability(req_a)
                            * self.location_b.request_poisson.probability(req_b)
                            * self.location_a.return_poisson.probability(ret_a)
                            * self.location_b.return_poisson.probability(ret_b)
                        )

                        valid_req_a = min(new_state[0], req_a)
                        valid_req_b = min(new_state[1], req_b)

                        rental_reward = (
                            valid_req_a + valid_req_b
                        ) * JackCarRentalParameters.RENTAL_CREDIT

                        next_state = [
                            max(
                                min(
                                    new_state[0] - valid_req_a + ret_a,
                                    JackCarRentalParameters.MAX_CARS,
                                ),
                                0,
                            ),
                            max(
                                min(
                                    new_state[1] - valid_req_b + ret_b,
                                    JackCarRentalParameters.MAX_CARS,
                                ),
                                0,
                            ),
                        ]

                        total_reward += scenario_prob * (
                            rental_reward
                            + JackCarRentalParameters.DISCOUNT_FACTOR
                            * self.value_function[next_state[0]][next_state[1]]
                        )

        return total_reward

    def policy_evaluation(self, threshold=0.01):
        
        while True:
            delta = 0
            for i in range(self.value_function.shape[0]):
                for j in range(self.value_function.shape[1]):
                    old_value = self.value_function[i][j]
                    self.value_function[i][j] = self.compute_expected_reward(
                        [i, j], self.policy[i][j]
                    )
                    delta = max(delta, abs(old_value - self.value_function[i][j]))

            if delta < threshold:
                break

    def policy_improvement(self):
        policy_stable = True

        for i in range(self.value_function.shape[0]):
            for j in range(self.value_function.shape[1]):
                old_action = self.policy[i][j]

                max_transfer_from_a = min(i, JackCarRentalParameters.MAX_TRANSFER)
                max_transfer_to_a = min(j, JackCarRentalParameters.MAX_TRANSFER)

                best_action = None
                max_value = float("-inf")

                for action in range(-max_transfer_to_a, max_transfer_from_a + 1):
                    value = self.compute_expected_reward([i, j], action)
                    if value > max_value:
                        max_value = value
                        best_action = action

                self.policy[i][j] = best_action

                if old_action != best_action:
                    policy_stable = False

        return policy_stable

    def solve(self):
        iterations = 0
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()

            self._visualize_policy_and_value(iterations)

            iterations += 1
            if policy_stable:
                break

        return self.policy, self.value_function

    def _visualize_policy_and_value(self, iteration):
        
        plt.figure(figsize=(12, 5))

        plt.subplot(121)
        sns.heatmap(self.policy, cmap="coolwarm", annot=True, fmt="d", cbar=False)
        plt.title(f"Policy (Iteration {iteration})")

        plt.subplot(122)
        sns.heatmap(self.value_function, cmap="viridis", cbar=True)
        plt.title(f"Value Function (Iteration {iteration})")

        plt.tight_layout()
        plt.savefig(f"jack_car_rental_iteration_{iteration}.png")
        plt.close()


def main():

    location_a = Location(request_rate=3, return_rate=3)
    location_b = Location(request_rate=4, return_rate=2)

    environment = JackCarRentalEnvironment(location_a, location_b)
    policy, value_function = environment.solve()

    print("Final Policy:")
    print(policy)
    print("Final Value Function:")
    print(value_function)


if __name__ == "__main__":
    main()
