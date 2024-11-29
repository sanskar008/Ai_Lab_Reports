import random
import itertools
import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = [" "] * 9
        self.current_player = "X"

    def make_move(self, position):
        if self.board[position] == " ":
            self.board[position] = self.current_player
            self.current_player = "O" if self.current_player == "X" else "X"
            return True
        return False

    def check_winner(self):

        wins = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]

        for combo in wins:
            if (
                self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]]
            ) and self.board[combo[0]] != " ":
                return self.board[combo[0]]

        if " " not in self.board:
            return "Tie"

        return None

    def __str__(self):
        return "\n".join(
            [
                f"{self.board[0]}|{self.board[1]}|{self.board[2]}",
                f"{self.board[3]}|{self.board[4]}|{self.board[5]}",
                f"{self.board[6]}|{self.board[7]}|{self.board[8]}",
            ]
        )


class MENACE:
    def __init__(self):

        self.matchboxes = {}
        self.history = []

    def board_to_state(self, board):
        return "".join(board)

    def get_possible_moves(self, board_state):
        return [str(i) for i, x in enumerate(board_state) if x == " "]

    def initialize_matchbox(self, board_state):
        if board_state not in self.matchboxes:
            moves = self.get_possible_moves(board_state)

            self.matchboxes[board_state] = {move: 3 for move in moves}

    def select_move(self, board_state):

        self.initialize_matchbox(board_state)

        moves = self.matchboxes[board_state]
        total_beads = sum(moves.values())

        if total_beads == 0:
            moves = {move: 1 for move in moves}
            total_beads = len(moves)

        r = random.randint(1, total_beads)
        cum = 0
        for move, beads in moves.items():
            cum += beads
            if r <= cum:

                self.history.append((board_state, move))
                return int(move)

    def update(self, result):

        reward_map = {
            "win": 3,
            "loss": -1,
            "draw": 1,
        }

        for board_state, move in self.history:

            self.initialize_matchbox(board_state)

            current_beads = self.matchboxes[board_state][str(move)]
            new_beads = max(0, current_beads + reward_map[result])
            self.matchboxes[board_state][str(move)] = new_beads

        self.history.clear()


def play_game(menace, human_player="O"):
    game = TicTacToe()

    while True:

        board_state = game.board_to_state()
        move = menace.select_move(board_state)
        game.make_move(move)

        print("MENACE's move:")
        print(game)

        winner = game.check_winner()
        if winner:
            if winner == "X":
                menace.update("win")
                print("MENACE wins!")
            elif winner == "Tie":
                menace.update("draw")
                print("It's a draw!")
            return winner

        while True:
            try:
                move = int(input("Enter your move (0-8): "))
                if game.make_move(move):
                    break
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Please enter a number between 0-8")

        print("Human's move:")
        print(game)

        winner = game.check_winner()
        if winner:
            if winner == "O":
                menace.update("loss")
                print("Human wins!")
            elif winner == "Tie":
                menace.update("draw")
                print("It's a draw!")
            return winner


def train_menace(menace, num_games=1000):
    for _ in range(num_games):
        play_game(menace)


menace = MENACE()
train_menace(menace, num_games=1000)


print("\nLet's play against the trained MENACE!")
play_game(menace)
