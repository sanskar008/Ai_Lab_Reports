import numpy as np


class hop_net:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def retrieve(self, pattern, max_iterations=100):
        state = pattern.copy()
        for _ in range(max_iterations):
            new_state = np.sign(self.weights @ state)
            new_state[new_state == 0] = 1
            if np.array_equal(state, new_state):
                break
            state = new_state
        return state


def gen_pat(num_patterns, size):
    return [np.random.choice([-1, 1], size=size) for _ in range(num_patterns)]


if __name__ == "__main__":
    size = 100
    num_patterns = 15
    patterns = gen_pat(num_patterns, size)

    hopfield = hop_net(size)
    hopfield.train(patterns)

    successes = 0
    for pattern in patterns:
        noisy_pattern = pattern.copy()
        noisy_pattern[:5] *= -1
        retrieved = hopfield.retrieve(noisy_pattern)
        if np.array_equal(retrieved, pattern):
            successes += 1

    print(f"Patterns retrieved successfully: {successes}/{num_patterns}")
