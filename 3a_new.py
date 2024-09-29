import heapq
import copy

class MarblePuzzle:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]

    def heuristic_1(self, state):
        return sum(row.count(1) for row in state)

    def heuristic_2(self, state):
        return sum(state[i][j] for i in range(7) for j in range(7) if state[i][j] == 1 and (i, j) != (3, 3))

    def get_neighbors(self, state):
        neighbors = []
        for i in range(7):
            for j in range(7):
                if state[i][j] == 1:  
                    for direction in self.directions:
                        ni, nj = i + direction[0], j + direction[1]
                        mid_i, mid_j = i + direction[0] // 2, j + direction[1] // 2
                        if 0 <= ni < 7 and 0 <= nj < 7:
                            if state[ni][nj] == 0 and state[mid_i][mid_j] == 1:
                                new_state = copy.deepcopy(state)
                                new_state[i][j] = 0      
                                new_state[mid_i][mid_j] = 0  
                                new_state[ni][nj] = 1     
                                neighbors.append((new_state, 1))  
        return neighbors

    def reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[tuple(map(tuple, current))]
        path.append(start)
        path.reverse()
        return path

    def a_star_search(self, heuristic):
        open_list = []
        heapq.heappush(open_list, (0 + heuristic(self.initial_state), 0, self.initial_state))
        came_from = {}
        g_score = {tuple(map(tuple, self.initial_state)): 0}
        while open_list:
            _, current_g, current_state = heapq.heappop(open_list)
            if current_state == self.goal_state:
                return self.reconstruct_path(came_from, self.initial_state, current_state)
            for neighbor, cost in self.get_neighbors(current_state):
                neighbor_tuple = tuple(map(tuple, neighbor))
                tentative_g_score = current_g + cost
                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)
                    heapq.heappush(open_list, (f_score, tentative_g_score, neighbor))
                    came_from[neighbor_tuple] = current_state
        return None  

    def best_first_search(self, heuristic):
        open_list = []
        heapq.heappush(open_list, (heuristic(self.initial_state), self.initial_state))
        came_from = {}
        visited = set()
        visited.add(tuple(map(tuple, self.initial_state)))
        while open_list:
            _, current_state = heapq.heappop(open_list)
            if current_state == self.goal_state:
                return self.reconstruct_path(came_from, self.initial_state, current_state)
            for neighbor, _ in self.get_neighbors(current_state):
                neighbor_tuple = tuple(map(tuple, neighbor))
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    heapq.heappush(open_list, (heuristic(neighbor), neighbor))
                    came_from[neighbor_tuple] = current_state
        return None  

    def uniform_cost_search(self):
        open_list = []
        heapq.heappush(open_list, (0, self.initial_state))
        came_from = {}
        g_score = {tuple(map(tuple, self.initial_state)): 0}
        while open_list:
            current_g, current_state = heapq.heappop(open_list)
            if current_state == self.goal_state:
                return self.reconstruct_path(came_from, self.initial_state, current_state)
            for neighbor, cost in self.get_neighbors(current_state):
                neighbor_tuple = tuple(map(tuple, neighbor))
                tentative_g_score = current_g + cost
                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    g_score[neighbor_tuple] = tentative_g_score
                    heapq.heappush(open_list, (tentative_g_score, neighbor))
                    came_from[neighbor_tuple] = current_state
        return None  

initial_state = [
    [-1, -1,  1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1, -1, -1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  0,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1],
    [-1, -1,  1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1, -1, -1],
]

goal_state = [
    [-1, -1,  0,  0,  0, -1, -1],
    [-1, -1,  0,  0,  0, -1, -1],
    [ 0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  1,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0],
    [-1, -1,  0,  0,  0, -1, -1],
    [-1, -1,  0,  0,  0, -1, -1],
]

puzzle = MarblePuzzle(initial_state, goal_state)
print("Running A* Search with Heuristic 1:")
a_star_path = puzzle.a_star_search(puzzle.heuristic_1)
if a_star_path:
    print(f"Solution found with {len(a_star_path) - 1} moves.")
else:
    print("No solution found.")

print("\nRunning Best-First Search with Heuristic 2:")
best_first_path = puzzle.best_first_search(puzzle.heuristic_2)
if best_first_path:
    print(f"Solution found with {len(best_first_path) - 1} moves.")
else:
    print("No solution found.")

print("\nRunning Uniform Cost Search:")
uniform_cost_path = puzzle.uniform_cost_search()
if uniform_cost_path:
    print(f"Solution found with {len(uniform_cost_path) - 1} moves.")
else:
    print("No solution found.")