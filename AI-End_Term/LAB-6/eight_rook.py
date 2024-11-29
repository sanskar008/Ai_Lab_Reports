import numpy as np


class RookHop:
    def __init__(self, size=8):
        self.size = size
        self.weights = np.full((size, size), -1)
        np.fill_diagonal(self.weights, 0)

    def update_state(self, current_state, iterations=10):
        state = np.copy(current_state)
        for _ in range(iterations):
            new_state = np.zeros(self.size)
            for i in range(self.size):
                activation = np.dot(self.weights[i], state)
                new_state[i] = 1 if activation > 0 else -1
            state = new_state
        return state


initial_state = np.random.choice([-1, 1], size=(8,))
eight_rook = RookHop(size=8)
final_state = eight_rook.update_state(initial_state)
print("Final State:", final_state)
