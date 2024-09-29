class GraphSearchAgent:
    def __init __(self, problem):
        self.problem = problem
        self.frontier = []  # The frontier is where we store the states to be explored (queue)
        self.explored_set = set()  # Explored_set is used to keep track of states that have already been explored (hash table)
        self.frontier.append(problem.initial_state)

    def search(self):
        while self.frontier:
            current_state = self.frontier.pop(0)  # Using pop(0) for a queue-like behavior
            if self.problem.is_goal_state(current_state):
                return self.reconstruct_path(current_state)
            self.explored_set.add(current_state)
            for action in self.problem.actions(current_state):
                next_state = self.problem.result(current_state, action)
                if next_state not in self.explored_set and next_state not in self.frontier:
                    self.frontier.append(next_state)
        return "failure"

    def reconstruct_path(self, current_state):
        # TO DO: implement path reconstruction
        pass


class BacktrackingPath:
    def __init__(self, source_state, goal_state):
        self.source_state = source_state
        self.goal_state = goal_state
        self.path = []

    def backtrack(self):
        current_state = self.goal_state
        while current_state != self.source_state:
            action = self.get_action_taken_to_reach(current_state)
            self.path.insert(0, action)
            current_state = self.get_parent_of(current_state)
        return self.path

    def get_action_taken_to_reach(self, current_state):
        # TO DO: implement action retrieval
        pass

    def get_parent_of(self, current_state):
        # TO DO: implement parent retrieval
        pass


class Puzzle8:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def generate_actions(self, state):
        actions = []
        empty_index = state.index('_')
        row, col = divmod(empty_index, 3)
        if row > 0:
            actions.append('Up')
        if row < 2:
            actions.append('Down')
        if col > 0:
            actions.append('Left')
        if col < 2:
            actions.append('Right')
        return actions

    def apply_action(self, state, action):
        new_state = list(state)
        empty_index = new_state.index('_')
        if action == 'Up':
            new_state[empty_index], new_state[empty_index - 3] = new_state[empty_index - 3], new_state[empty_index]
        elif action == 'Down':
            new_state[empty_index], new_state[empty_index + 3] = new_state[empty_index + 3], new_state[empty_index]
        elif action == 'Left':
            new_state[empty_index], new_state[empty_index - 1] = new_state[empty_index - 1], new_state[empty_index]
        elif action == 'Right':
            new_state[empty_index], new_state[empty_index + 1] = new_state[empty_index + 1], new_state[empty_index]
        return tuple(new_state)

    def solve(self, max_depth):
        agent = GraphSearchAgent(self)
        solution_path, _, _, _, _ = agent.search()
        if solution_path:
            self.visualize_solution(self.initial_state, solution_path)
        else:
            print("No solution found.")

    def visualize_solution(self, initial_state, solution_path):
        # TO DO: implement visualization
        pass


def get_memory_usage():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def main():
    initial_state = ('_', 1, 2, 3, 4, 5, 7, 8, 6)
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, '_')
    max_depth = 16
    puzzle = Puzzle8(initial_state, goal_state)
    puzzle.solve(max_depth)
    print("Depth | Time Elapsed (s) | Memory Used (bytes)")
    print("------|------------------|-------------------")
    for depth, time_elapsed, memory_used in [(1, 1, get_memory_usage())]:
        print(f"{depth:<6}|{time_elapsed:<18}|{memory_used}")


if __name__ == "__main__":
    main()