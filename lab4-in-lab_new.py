from collections import deque

class PuzzleSolver:
    def __init__(self, initial_state, target_state):
        self.initial_state = initial_state
        self.target_state = target_state

    def is_valid_transition(self, previous, new):
        # Check if the transition to the new state is valid
        prev_zero_idx = previous.index(0)
        new_zero_idx = new.index(0)
        
        if abs(prev_zero_idx - new_zero_idx) > 2:
            return False
        if new_zero_idx < 0 or new_zero_idx >= len(previous):
            return False
        if (previous[new_zero_idx] == -1 and prev_zero_idx <= new_zero_idx) or \
           (previous[new_zero_idx] == 1 and prev_zero_idx >= new_zero_idx):
            return False
        return True

    def generate_successors(self, state):
        # Generate all possible valid successor states
        moves = [-2, -1, 1, 2]
        successors = []
        zero_idx = state.index(0)
        
        for move in moves:
            new_idx = zero_idx + move
            if 0 <= new_idx < len(state):
                new_state = list(state)
                new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
                new_state_tuple = tuple(new_state)
                if self.is_valid_transition(state, new_state_tuple):
                    successors.append(new_state_tuple)
        return successors

    def bfs(self):
        # Perform BFS to find the path from initial_state to target_state
        queue = deque([(self.initial_state, [])])
        visited = set()
        
        while queue:
            current_state, path = queue.popleft()
            
            if current_state in visited:
                continue
            
            visited.add(current_state)
            for successor in self.generate_successors(current_state):
                if successor == self.target_state:
                    return path + [current_state, successor]
                queue.append((successor, path + [current_state]))
        
        return None

def main():
    start_state = (-1, -1, -1, 0, 1, 1, 1)
    goal_state = (1, 1, 1, 0, -1, -1, -1)
    
    solver = PuzzleSolver(start_state, goal_state)
    solution_path = solver.bfs()
    
    if solution_path:
        for state in solution_path:
            print(state)
    else:
        print("No solution found")

if __name__ == "__main__":
    main()