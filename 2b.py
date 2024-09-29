from collections import deque

class State:
    def __init__(self, state):
        self.state = state

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def __repr__(self):
        return str(self.state)

class ProblemSolver:
    def __init__(self, initial_state, target_state):
        self.initial_state = State(initial_state)
        self.target_state = State(target_state)

    def is_valid_transition(self, current_state, next_state):
        current_zero_idx = current_state.state.index(0)
        next_zero_idx = next_state.state.index(0)
        
        if abs(current_zero_idx - next_zero_idx) > 2:
            return False
        if next_zero_idx < 0 or next_zero_idx >= len(current_state.state):
            return False
        if (current_state.state[next_zero_idx] == -1 and current_zero_idx <= next_zero_idx) or \
           (current_state.state[next_zero_idx] == 1 and current_zero_idx >= next_zero_idx):
            return False
        return True

    def generate_successors(self, state):
        possible_moves = [-2, -1, 1, 2]
        successors = []
        zero_index = state.state.index(0)
        
        for move in possible_moves:
            new_index = zero_index + move
            if 0 <= new_index < len(state.state):
                new_state = list(state.state)
                new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
                new_state_tuple = tuple(new_state)
                new_state = State(new_state_tuple)
                if self.is_valid_transition(state, new_state):
                    successors.append(new_state)
        return successors

    def bfs(self):
        queue = deque([(self.initial_state, [])])
        explored = set()
        
        while queue:
            current_state, path = queue.popleft()
            
            if current_state in explored:
                continue
            
            explored.add(current_state)
            for successor in self.generate_successors(current_state):
                if successor == self.target_state:
                    return path + [current_state.state, successor.state]
                queue.append((successor, path + [current_state.state]))
        
        return None

def main():
    initial_state = (-1, -1, -1, 0, 1, 1, 1)
    target_state = (1, 1, 1, 0, -1, -1, -1)
    
    solver = ProblemSolver(initial_state, target_state)
    solution_path = solver.bfs()
    
    if solution_path:
        for state in solution_path:
            print(state)
    else:
        print("No solution found")

if __name__ == "__main__":
    main()