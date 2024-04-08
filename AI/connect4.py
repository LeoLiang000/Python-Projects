import numpy as np
import math

"""
reference: https://github.com/KeithGalli/Connect4-Python/blob/master/connect4.py
"""


class ConnectFour:
    def __init__(self, row_count=6, column_count=7):
        self.ROW_COUNT = row_count
        self.COLUMN_COUNT = column_count
        self.board = self.create_board()
        self.game_over = False
        self.turn = 0  # 0 for Player 1; 1 for Player 2

    def create_board(self):
        return np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def is_valid_location(self, col):
        return self.board[self.ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(self.ROW_COUNT):
            if self.board[r][col] == 0:
                return r

    def print_board(self):
        print(np.flip(self.board, 0))

    def winning_move(self, piece):
        # Horizontal win check
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT):
                if self.board[r][c] == piece and self.board[r][c + 1] == piece and \
                        self.board[r][c + 2] == piece and self.board[r][c + 3] == piece:
                    # print("Horizontal win")
                    return True

        # Vertical win check
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c] == piece and \
                        self.board[r + 2][c] == piece and self.board[r + 3][c] == piece:
                    # print("Vertical win")
                    return True

        # Positive diagonal win check
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and \
                        self.board[r + 2][c + 2] == piece and self.board[r + 3][c + 3] == piece:
                    # print("Positive diagonal win")
                    return True

        # Negative diagonal win check
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(3, self.ROW_COUNT):
                if self.board[r][c] == piece and self.board[r - 1][c + 1] == piece and \
                        self.board[r - 2][c + 2] == piece and self.board[r - 3][c + 3] == piece:
                    # print("Negative diagonal win")
                    return True

    def is_game_over(self):
        # Game is over if there is a win or the board is full (draw)
        return self.winning_move(1) or self.winning_move(2) or not any(
            self.is_valid_location(c) for c in range(self.COLUMN_COUNT))

    def evaluate_score(self):
        return np.count_nonzero(self.board == 2) - np.count_nonzero(self.board == 1)

    def get_available_moves(self):
        return [c for c in range(self.COLUMN_COUNT) if self.is_valid_location(c)]

    def make_move(self, col, piece):
        if self.is_valid_location(col):
            row = self.get_next_open_row(col)
            if row is not None:
                self.drop_piece(row, col, piece)
                return True

        return False

    def undo_move(self, col):
        for r in range(self.ROW_COUNT - 1, -1, -1):
            if self.board[r][col] != 0:
                self.board[r][col] = 0
                break

