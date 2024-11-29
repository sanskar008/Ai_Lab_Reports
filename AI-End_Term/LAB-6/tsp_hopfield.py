import numpy as np


class tsp_hop:
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.neurons = num_cities * num_cities
        self.weights = np.zeros((self.neurons, self.neurons))
        self.distance_matrix = None

    def weight_init(self, distances, A=500, B=500, C=200, D=200):
        self.distance_matrix = distances
        N = self.num_cities

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        idx_1 = i * N + j
                        idx_2 = k * N + l
                        if i == k and j != l:
                            self.weights[idx_1][idx_2] -= A
                        if i != k and j == l:
                            self.weights[idx_1][idx_2] -= B
                        if i == k and j == l:
                            self.weights[idx_1][idx_2] -= C
                        if (i + 1) % N == k:
                            self.weights[idx_1][idx_2] -= D * distances[j][l]

    def optimize(self, initial_state, steps=100):
        state = np.copy(initial_state)
        for _ in range(steps):
            activations = np.dot(self.weights, state)
            state = np.where(activations > 0, 1, 0)
        return state

    def calculate_tour_length(self, tour):
        tour_matrix = tour.reshape((self.num_cities, self.num_cities))
        total_distance = 0
        for i in range(self.num_cities):
            current_city = np.argmax(tour_matrix[i])
            next_city = np.argmax(tour_matrix[(i + 1) % self.num_cities])
            total_distance += self.distance_matrix[current_city][next_city]
        return total_distance


def generate_distance_matrix(num_cities, min_dist=1, max_dist=100):
    distances = np.random.randint(min_dist, max_dist, size=(num_cities, num_cities))
    np.fill_diagonal(distances, 0)
    return distances


num_cities = 10
distances = generate_distance_matrix(num_cities)
tsp = tsp_hop(num_cities)
tsp.weight_init(distances)

initial_state = np.random.choice([0, 1], size=(num_cities * num_cities))
final_state = tsp.optimize(initial_state)

print("Optimized State:", final_state)
print("Tour Length:", tsp.calculate_tour_length(final_state))
