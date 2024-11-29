import numpy as np


class HopfieldNet:
    def __init__(self, size):
        self.size = size
        self.connection_matrix = np.zeros((size, size))

    def learn_patterns(self, training_data):
        for pattern in training_data:
            self.connection_matrix += np.outer(pattern, pattern)
        np.fill_diagonal(self.connection_matrix, 0)
        self.connection_matrix /= len(training_data)

    def reconstruct(self, input_pattern, max_iterations=15):
        current_state = np.copy(input_pattern)
        for _ in range(max_iterations):
            updated_state = np.zeros_like(current_state)
            for i in range(self.size):
                net_input = np.dot(self.connection_matrix[i], current_state)
                updated_state[i] = 1 if net_input > 0 else -1

            if np.array_equal(current_state, updated_state):
                break

            current_state = updated_state

        return current_state


training_patterns = [np.array([1, -1, 1, -1, 1]), np.array([-1, 1, -1, 1, -1])]
test_pattern = np.array([1, -1, -1, -1, 1])

network = HopfieldNet(size=5)
network.learn_patterns(training_patterns)

output = network.reconstruct(test_pattern)
print("Reconstructed Pattern:", output)
