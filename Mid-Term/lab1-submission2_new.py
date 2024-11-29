from collections import deque

class PuzzleSolver:
    def _init_(self, start, goal):
        self.start_state = start
        self.goal_state = goal

    def is_valid_transition(self, current, new):
        current_zero_idx = current.index(0)
        new_zero_idx = new.index(0)
        
        if abs(current_zero_idx - new_zero_idx) > 2:
            return False
        if new_zero_idx < 0 or new_zero_idx >= len(current):
            return False
        if (current[new_zero_idx] == -1 and current_zero_idx <= new_zero_idx) or \
           (current[new_zero_idx] == 1 and current_zero_idx >= new_zero_idx):
            return False
        return True

    def generate_successors(self, state):
        moves = [-2, -1, 1, 2]
        zero_pos = state.index(0)
        successors = []

        for move in moves:
            new_pos = zero_pos + move
            if 0 <= new_pos < len(state):
                new_state = list(state)
                new_state[zero_pos], new_state[new_pos] = new_state[new_pos], new_state[zero_pos]
                new_state_tuple = tuple(new_state)
                if self.is_valid_transition(state, new_state_tuple):
                    successors.append(new_state_tuple)
        return successors

    def bfs(self):
        queue = deque([(self.start_state, [])])
        visited = set()

        while queue:
            current_state, path = queue.popleft()

            if current_state in visited:
                continue

            visited.add(current_state)
            successors = self.generate_successors(current_state)

            for successor in successors:
                if successor == self.goal_state:
                    return path + [current_state, successor]
                queue.append((successor, path + [current_state]))

        return None

def main():
    initial_state = (-1, -1, -1, 0, 1, 1, 1)
    target_state = (1, 1, 1, 0, -1, -1, -1)

    solver = PuzzleSolver(initial_state, target_state)
    solution_path = solver.bfs()

    if solution_path:
        for state in solution_path:
            print(state)
    else:
        print("No solution found")

if _name_ == "_main_":
    main()