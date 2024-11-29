import numpy as np
import matplotlib.pyplot as plt
import random
import copy

# Load the scrambled image from the uploaded file
import scipy.io as sio
scrambled = sio.loadmat('/mnt/data/scrambled_lena.mat')['scrambled']

# Display the scrambled image
plt.imshow(scrambled, cmap='gray')
plt.title("Scrambled Image")
plt.show()

# Helper function to calculate the difference between two adjacent tiles
def calculate_difference(tile1, tile2):
    return np.sum(np.abs(tile1 - tile2))

# Helper function to calculate the total cost of the current arrangement
def calculate_cost(image_tiles, grid_size):
    total_cost = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if j < grid_size - 1:
                total_cost += calculate_difference(image_tiles[i][j][:, -1], image_tiles[i][j+1][:, 0])
            if i < grid_size - 1:
                total_cost += calculate_difference(image_tiles[i][j][-1, :], image_tiles[i+1][j][0, :])
    return total_cost

# Helper function to swap two random tiles
def swap_tiles(image_tiles, grid_size):
    i1, j1 = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
    i2, j2 = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
    new_tiles = copy.deepcopy(image_tiles)
    new_tiles[i1][j1], new_tiles[i2][j2] = new_tiles[i2][j2], new_tiles[i1][j1]
    return new_tiles

# Simulated Annealing algorithm
def simulated_annealing(scrambled, grid_size, initial_temp, cooling_rate, max_iterations):
    # Split the scrambled image into tiles
    tile_height = scrambled.shape[0] // grid_size
    tile_width = scrambled.shape[1] // grid_size
    image_tiles = [[scrambled[i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width]
                    for j in range(grid_size)] for i in range(grid_size)]
    
    current_state = image_tiles
    current_cost = calculate_cost(current_state, grid_size)
    temperature = initial_temp

    for iteration in range(max_iterations):
        # Generate a new candidate state by swapping two tiles
        new_state = swap_tiles(current_state, grid_size)
        new_cost = calculate_cost(new_state, grid_size)
        
        # Accept the new state with a certain probability
        if new_cost < current_cost:
            current_state, current_cost = new_state, new_cost
        else:
            probability = np.exp((current_cost - new_cost) / temperature)
            if random.random() < probability:
                current_state, current_cost = new_state, new_cost
        
        # Decrease the temperature
        temperature *= cooling_rate

        # Optional: print progress
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, Current Cost: {current_cost}")

    return current_state

# Parameters for Simulated Annealing
grid_size = 3  # Assuming a 3x3 grid
initial_temp = 1000
cooling_rate = 0.995
max_iterations = 10000

# Run the simulated annealing algorithm
solved_tiles = simulated_annealing(scrambled, grid_size, initial_temp, cooling_rate, max_iterations)

# Reconstruct the solved image
solved_image = np.block(solved_tiles)

# Display the solved image
plt.imshow(solved_image, cmap='gray')
plt.title("Solved Image")
plt.show()
