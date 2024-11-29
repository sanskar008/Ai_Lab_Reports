import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
from functools import lru_cache
from typing import Dict, Tuple, List


class JackCarRentalConfig:
    """Comprehensive configuration for the problem."""

    MAX_CARS = 20
    DISCOUNT_FACTOR = 0.9
    RENTAL_CREDIT = 10
    MOVING_COST = -2
    CONVERGENCE_THRESHOLD = 1e-4
    MAX_TRANSFER = 5
    REQUEST_RATE_A = 3
    RETURN_RATE_A = 3
    REQUEST_RATE_B = 4
    RETURN_RATE_B = 2


class PoissonProbabilityCache:
    """Efficient Poisson distribution probability calculator with caching."""

    def __init__(self, λ: float, epsilon: float = 1e-4):
        self.λ = λ
        self.probabilities = self._compute_probabilities(epsilon)

    def _compute_probabilities(self, epsilon: float) -> Dict[int, float]:
       
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

    @lru_cache(maxsize=None)
    def probability(self, n: int) -> float:
       return self.probabilities.get(n, 0)


class Location:
  
    def __init__(self, request_rate: float, return_rate: float):
        self.request_poisson = PoissonProbabilityCache(request_rate)
        self.return_poisson = PoissonProbabilityCache(return_rate)


class JackCarRentalEnvironment:
   
    def __init__(
        self,
        location_a: Location,
        location_b: Location,
        config: JackCarRentalConfig = None,
    ):
        self.config = config or JackCarRentalConfig()
        self.location_a = location_a
        self.location_b = location_b

        self.value_function = np.zeros(
            (self.config.MAX_CARS + 1, self.config.MAX_CARS + 1)
        )
        self.policy = np.zeros_like(self.value_function, dtype=int)

    def compute_expected_reward(self, state: Tuple[int, int], action: int) -> float:
       
        new_state = [
            max(min(state[0] - action, self.config.MAX_CARS), 0),
            max(min(state[1] + action, self.config.MAX_CARS), 0),
        ]

        total_reward = self.config.MOVING_COST * abs(action)

        req_probs_a = list(self.location_a.request_poisson.probabilities.items())
        req_probs_b = list(self.location_b.request_poisson.probabilities.items())
        ret_probs_a = list(self.location_a.return_poisson.probabilities.items())
        ret_probs_b = list(self.location_b.return_poisson.probabilities.items())

        for req_a, p_req_a in req_probs_a:
            for req_b, p_req_b in req_probs_b:
                for ret_a, p_ret_a in ret_probs_a:
                    for ret_b, p_ret_b in ret_probs_b:

                        scenario_prob = p_req_a * p_req_b * p_ret_a * p_ret_b

                        valid_req_a = min(new_state[0], req_a)
                        valid_req_b = min(new_state[1], req_b)

                        rental_reward = (
                            valid_req_a + valid_req_b
                        ) * self.config.RENTAL_CREDIT

                        next_state = [
                            max(
                                min(
                                    new_state[0] - valid_req_a + ret_a,
                                    self.config.MAX_CARS,
                                ),
                                0,
                            ),
                            max(
                                min(
                                    new_state[1] - valid_req_b + ret_b,
                                    self.config.MAX_CARS,
                                ),
                                0,
                            ),
                        ]

                        total_reward += scenario_prob * (
                            rental_reward
                            + self.config.DISCOUNT_FACTOR
                            * self.value_function[next_state[0]][next_state[1]]
                        )

        return total_reward

    def policy_evaluation(self, threshold: float = None) -> None:
       
        threshold = threshold or self.config.CONVERGENCE_THRESHOLD

        while True:
            delta = 0
            for i in range(self.value_function.shape[0]):
                for j in range(self.value_function.shape[1]):
                    old_value = self.value_function[i][j]
                    self.value_function[i][j] = self.compute_expected_reward(
                        (i, j), self.policy[i][j]
                    )
                    delta = max(delta, abs(old_value - self.value_function[i][j]))

            if delta < threshold:
                break

    def policy_improvement(self) -> bool:
      
        policy_stable = True

        for i in range(self.value_function.shape[0]):
            for j in range(self.value_function.shape[1]):
                old_action = self.policy[i][j]

                max_transfer_from_a = min(i, self.config.MAX_TRANSFER)
                max_transfer_to_a = min(j, self.config.MAX_TRANSFER)

                best_action = max(
                    range(-max_transfer_to_a, max_transfer_from_a + 1),
                    key=lambda action: self.compute_expected_reward((i, j), action),
                )

                self.policy[i][j] = best_action

                if old_action != best_action:
                    policy_stable = False

        return policy_stable

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
       
        iterations = 0
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()

            self._visualize(iterations)

            iterations += 1
            if policy_stable:
                break

        return self.policy, self.value_function

    def _visualize(self, iteration: int) -> None:
        
        plt.figure(figsize=(15, 6))

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
