import math
import random
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

from tic_tac_toe import TicTacToe
from connect4 import ConnectFour
from collections import defaultdict

TIC_AI = 'X'
TIC_OPPONENT = 'O'
TIC_DRAW = 'None'


# Connect Four integrated of default opponent with basic strategy: win move and block move
class SmartConnect4(ConnectFour):
    def __init__(self):
        super().__init__()
        self.players = {0: "AI",
                        1: "Opponent"}
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.current_winner = None

    def checkBlockMove(self, piece_opponent):
        # First, check if the Minimax AI can win in the next move and block that move
        for column in range(self.COLUMN_COUNT):
            if self.is_valid_location(column):
                next_row = self.get_next_open_row(column)
                self.board[next_row][column] = piece_opponent
                if self.winning_move(piece_opponent):
                    self.board[next_row][column] = 0
                    return column
                self.board[next_row][column] = 0
        return None

    def defaultOpponentMove(self, piece):
        piece_opponent = 1 if piece == 2 else 2
        column_block = self.checkBlockMove(piece_opponent)
        if column_block is not None: return column_block

        if self.is_valid_location(self.COLUMN_COUNT // 2): return self.COLUMN_COUNT // 2

        return random.choice([col for col in range(self.COLUMN_COUNT) if self.is_valid_location(col)])

    def playGameMinimax(self, isPruning=False):
        while not self.game_over:
            if self.turn == 0:  # Minimax AI turn
                print("Minimax AI makes moving...")
                move_block = self.checkBlockMove(2)  # check if block move exists
                if move_block is not None:
                    self.make_move(move_block, 1)
                else:
                    move, _ = minimax(self, 5, True, isTicTac=False, isPruning=isPruning)
                    self.make_move(move, 1)

            else:  # Default opponent turn
                print("Default opponent makes moving...")
                self.make_move(self.defaultOpponentMove(2), 2)

            self.print_board()

            if self.is_game_over():  # check game over
                if self.winning_move(self.turn + 1):
                    print(f"Player {self.players[self.turn]} wins!")
                    if self.players[self.turn] == 'AI':
                        return 'win'
                    elif self.players[self.turn] == 'Opponent':
                        return 'loss'
                else:
                    print("Game Over! It's a draw.")
                    return 'draw'

                self.game_over = True
            else:
                self.turn = (self.turn + 1) % 2  # Switch turns

    def playGameQLearning(self, epsilon=0.5, alpha=0.2, gamma=0.8, episodes=100):
        epsilon_min = 0.01
        epsilon_decay_factor = 0.995

        results = {'win': 0, 'loss': 0, 'draw': 0}
        result_trend = []

        for i in range(episodes):
            self.board = self.create_board()
            self.game_over = False

            while not self.game_over:
                # if self.turn == 0:
                #     print("Q-Learning AI is making a move...")
                #     move_block = self.checkBlockMove(2)  # check if block move exists
                #     if move_block is not None:
                #         self.make_move(move_block, 1)
                #     else:
                #         q_learning(self, self.q_table, self.get_state(), 1, epsilon, alpha, gamma, isConnect4=True)
                #
                # else:
                #     print("Default opponent is making a move...")
                #     self.make_move(self.defaultOpponentMove(2), 2)

                if self.turn == 0:
                    print("Default opponent is making a move...")
                    self.make_move(self.defaultOpponentMove(2), 2)

                else:
                    print("Q-Learning AI  is making a move...")
                    move_block = self.checkBlockMove(2)  # check if block move exists
                    if move_block is not None:
                        self.make_move(move_block, 1)
                    else:
                        q_learning(self, self.q_table, self.get_state(), 1, epsilon, alpha, gamma, isConnect4=True)

                self.print_board()

                if self.is_game_over():
                    # if self.winning_move(self.turn + 1):
                    #     print(f"Player {self.players[self.turn]} wins!")
                    #
                    #     if self.players[self.turn] == 'AI':
                    #         results['win'] += 1
                    #         result_trend.append('win')
                    #
                    #     elif self.players[self.turn] == 'Opponent':
                    #         results['loss'] += 1
                    #         result_trend.append('loss')
                    #
                    # else:
                    #     print("Game Over! It's a draw.")
                    #     results['draw'] += 1
                    #     result_trend.append('draw')

                    if self.winning_move(1):  # Q learning
                        print(f'Q learning wins')
                        results['win'] += 1
                        result_trend.append('win')

                    elif self.winning_move(2):  # default opponent
                        print(f'Opponent wins')
                        results['loss'] += 1
                        result_trend.append('loss')

                    else:
                        print(f'it is a draw')
                        results['draw'] += 1
                        result_trend.append('draw')

                    break

                self.turn = (self.turn + 1) % 2

            epsilon = max(epsilon_min, epsilon_decay_factor * epsilon)

        return results, result_trend

    def playGameC4MinimaxAgainstQlearning(self, isPruning=False, epsilon=0.1, alpha=0.1, gamma=0.8, episodes=100):
        epsilon_min = 0.01
        epsilon_decay_factor = 0.995

        results = {'win': 0, 'loss': 0, 'draw': 0}
        result_trend = []

        players = {0: 'QLearning',
                   1: 'Minimax'}

        for i in range(episodes):
            self.board = self.create_board()
            self.game_over = False

            while not self.game_over:
                # if self.turn == 0:
                #     print("Q-Learning AI is making a move...")
                #     # q_learning(self, self.q_table, self.get_state(), 1, epsilon, alpha, gamma, isConnect4=True)
                #
                #     move_block = self.checkBlockMove(2)  # check if block move exists
                #     if move_block is not None:
                #         self.make_move(move_block, 1)
                #     else:
                #         q_learning(self, self.q_table, self.get_state(), 1, epsilon, alpha, gamma, isConnect4=True)
                #
                # else:
                #     print("Minimax AI is making a move...")
                #     move_block = self.checkBlockMove(1)  # check if block move exists
                #     if move_block is not None:
                #         self.make_move(move_block, 2)
                #     else:
                #         col, _ = minimax(self, 5, True, isTicTac=False, isPruning=isPruning)
                #         self.make_move(col, 2)

                if self.turn == 0:
                    print("Minimax AI is making a move...")
                    move_block = self.checkBlockMove(1)  # check if block move exists
                    if move_block is not None:
                        self.make_move(move_block, 2)
                    else:
                        col, _ = minimax(self, 5, True, isTicTac=False, isPruning=isPruning)
                        self.make_move(col, 2)

                else:
                    print("Q-Learning AI is making a move...")
                    move_block = self.checkBlockMove(2)  # check if block move exists
                    if move_block is not None:
                        self.make_move(move_block, 1)
                    else:
                        q_learning(self, self.q_table, self.get_state(), 1, epsilon, alpha, gamma, isConnect4=True)

                # move_block = self.checkBlockMove(1)  # check if block move exists
                # if move_block is not None:
                #     self.make_move(move_block, 2)
                # else:
                #     col, _ = minimax(self, 5, True, isTicTac=False, isPruning=isPruning)
                #     if col is not None:
                #         print(f'Minimax best move: {col}')
                #         self.make_move(col, 2)

                self.print_board()

                if self.is_game_over():
                    if self.winning_move(1):  # Q learning
                        print(f"Player Q learning wins!")
                        results['win'] += 1
                        result_trend.append('win')

                    elif self.winning_move(2):  # minimax
                        print(f"Player minimax wins!")
                        results['loss'] += 1
                        result_trend.append('loss')

                    else:
                        print("Game Over! It's a draw.")
                        results['draw'] += 1
                        result_trend.append('draw')

                    break

                self.turn = (self.turn + 1) % 2

            epsilon = max(epsilon_min, epsilon_decay_factor * epsilon)

        return results, result_trend

    def get_state(self):
        return tuple(tuple(int(cell) for cell in row) for row in self.board)

    def evaluate_score_complex(self):
        score = 0
        if self.winning_move(1):
            score += 1000
        elif self.winning_move(2):
            score -= 1000

        score += self.check_threats(1) * 100  # Reward for player 1 creating opportunities
        score -= self.check_threats(2) * 100  # Penalize for player 2 creating opportunities

        return score

    def check_threats(self, player):
        threats = 0
        # Horizontal threats
        for r in range(self.ROW_COUNT):
            for c in range(self.COLUMN_COUNT - 3):
                window = [self.board[r][c + i] for i in range(4)]
                if window.count(player) == 3 and window.count(0) == 1:
                    threats += 1

        # Vertical threats
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT - 3):
                window = [self.board[r + i][c] for i in range(4)]
                if window.count(player) == 3 and window.count(0) == 1:
                    threats += 1

        # Positive diagonal threats
        for r in range(self.ROW_COUNT - 3):
            for c in range(self.COLUMN_COUNT - 3):
                window = [self.board[r + i][c + i] for i in range(4)]
                if window.count(player) == 3 and window.count(0) == 1:
                    threats += 1

        # Negative diagonal threats
        for r in range(3, self.ROW_COUNT):
            for c in range(self.COLUMN_COUNT - 3):
                window = [self.board[r - i][c + i] for i in range(4)]
                if window.count(player) == 3 and window.count(0) == 1:
                    threats += 1

        return threats


# Tic Tac Toe integrated of default opponent with basic strategy: win move and block move
class SmartTicTacToe(TicTacToe):
    def __init__(self, letter):
        super().__init__()
        self.letter = letter
        self.letter_opponent = 'O' if letter == 'X' else 'X'
        self.turn = 0  # 0 for Player 1; 1 for Player 2
        self.players = {'X': "AI",
                        'O': "Default Opponent"}
        self.q_table = defaultdict(lambda: defaultdict(float))

    def isWinningMove(self, square, letter):
        self.board[square] = letter
        isWinning = self.winner(square, letter)
        self.board[square] = ' '  # Undo the move
        return isWinning

    def getMove(self):
        best_move = None
        for move in self.get_available_moves():
            # check win move
            if self.isWinningMove(move, self.letter):
                best_move = move
                break

            # check block move
            if not best_move and self.isWinningMove(move, self.letter_opponent):
                best_move = move

        if best_move is not None:
            return best_move
        else:
            return random.choice(self.get_available_moves())

    def playGameMinimax(self, isPruning=False, playTimes=1):
        result_trend = ''

        while not self.is_game_over():
            # if self.turn == 0:
            #     print("Minimax AI makes moving...")
            #     move, _ = minimax(self, depth=6, isMaxPlayer=True, isPruning=isPruning, isTicTac=True)
            #     self.make_move(move, self.letter)
            # else:
            #     print("Default opponent's turn:")
            #     self.make_move(self.getMove(), self.letter_opponent)

            if self.turn == 0:
                print("Default opponent makes moving...")
                self.make_move(self.getMove(), self.letter_opponent)

            else:
                print("Minimax AI makes moving:")
                move, _ = minimax(self, depth=6, isMaxPlayer=False, isPruning=isPruning, isTicTac=True)
                self.make_move(move, self.letter)

            self.print_board()

            if self.is_game_over():
                if self.current_winner:
                    print(f"Player {self.players[self.current_winner]} wins!")
                    if self.current_winner == self.letter:
                        result_trend = 'win'

                    elif self.current_winner == self.letter_opponent:
                        result_trend = 'loss'
                else:
                    print("Game over! It's a tie.")
                    result_trend = 'draw'

                break  # Exit the loop if the game is over

            self.turn = (self.turn + 1) % 2  # Switch turns

        return result_trend

    def playGameQLearning(self, epsilon=0.1, alpha=0.1, gamma=0.9, episodes=5000):
        epsilon_min = 0.01
        epsilon_decay_factor = 0.995

        results = {'win': 0, 'loss': 0, 'draw': 0}
        result_trend = []

        for i in range(episodes):
            self.board = self.make_board()
            self.current_winner = None
            self.turn = 0

            while not self.is_game_over():
                # if self.turn == 0:
                #     print("QLearning AI makes moving...")
                #     move_block = self.check_block_move(self.letter)
                #     if move_block is not None:
                #         self.make_move(move_block, self.letter)
                #     else:
                #         q_learning(self, self.q_table, tuple(self.board), self.letter, epsilon, alpha, gamma)
                #
                # else:
                #     print("Default opponent's turn:")
                #     move_block = self.check_block_move(self.letter_opponent)
                #     if move_block is not None:
                #         self.make_move(move_block, self.letter_opponent)
                #     else:
                #         self.make_move(self.getMove(), self.letter_opponent)

                if self.turn == 0:
                    print("Default opponent makes moving...")
                    move_block = self.check_block_move(self.letter_opponent)
                    if move_block is not None:
                        self.make_move(move_block, self.letter_opponent)
                    else:
                        self.make_move(self.getMove(), self.letter_opponent)
                else:
                    print("QLearning AI makes moving:")
                    move_block = self.check_block_move(self.letter)
                    if move_block is not None:
                        self.make_move(move_block, self.letter)
                    else:
                        q_learning(self, self.q_table, tuple(self.board), self.letter, epsilon, alpha, gamma)

                self.print_board()

                if self.is_game_over():
                    if self.current_winner:
                        print(f"Player {self.players[self.current_winner]} wins!")
                        if self.current_winner == self.letter:
                            results['win'] += 1
                            result_trend.append('win')

                        elif self.current_winner == self.letter_opponent:
                            results['loss'] += 1
                            result_trend.append('loss')

                    else:
                        print("Game over! It's a tie.")
                        results['draw'] += 1
                        result_trend.append('draw')

                    break

                self.turn = (self.turn + 1) % 2

            epsilon = max(epsilon_min, epsilon_decay_factor * epsilon)

        return results, result_trend

    def playTicTacToeMinimaxAgainstQlearning(self, isPruning=False, epsilon=0.1, alpha=0.1, gamma=0.9, episodes=5000):
        epsilon_min = 0.01
        epsilon_decay_factor = 0.995

        results = {'win': 0, 'loss': 0, 'draw': 0}
        result_trend = []

        players = {'X': "Qlearning",
                   'O': "Minimax"}

        for i in range(episodes):
            self.board = self.make_board()
            self.current_winner = None
            self.turn = 0

            while not self.is_game_over():
                if self.turn == 0:  # QLearning: label X, Minimax: label O
                    print("QLearning AI makes moving...")
                    q_learning(self, self.q_table, tuple(self.board), self.letter, epsilon, alpha, gamma)

                else:
                    print("Minimax AI makes moving:")

                    move, _ = minimax(self, depth=6, isMaxPlayer=False, isPruning=isPruning, isTicTac=True)
                    if move is not None:
                        self.make_move(move, self.letter_opponent)

                # if self.turn == 0:  # QLearning: label X, Minimax: label O
                #     print("Minimax AI makes moving...")
                #     move, _ = minimax(self, depth=6, isMaxPlayer=True, isPruning=isPruning, isTicTac=True)
                #     if move is not None:
                #         self.make_move(move, self.letter_opponent)
                #
                # else:
                #     print("QLearning AI makes moving:")
                #     q_learning(self, self.q_table, tuple(self.board), self.letter, epsilon, alpha, gamma)

                self.print_board()

                if self.is_game_over():
                    if self.current_winner:
                        if self.current_winner == self.letter:  # label 'X' for Q learning
                            print(f"Player {players[self.current_winner]} wins!")
                            results['win'] += 1
                            result_trend.append('win')

                        elif self.current_winner == self.letter_opponent:
                            print(f"Player {players[self.current_winner]} wins!")
                            results['loss'] += 1
                            result_trend.append('loss')

                    else:
                        print("Game over! It's a tie.")
                        results['draw'] += 1
                        result_trend.append('draw')

                    break

                self.turn = (self.turn + 1) % 2

            epsilon = max(epsilon_min, epsilon_decay_factor * epsilon)

        return results, result_trend

    def evaluate_score_complex(self, player):
        score = 0
        if self.current_winner == player:
            return 100
        elif self.current_winner and self.current_winner != player:
            return -100
        lines = self.get_lines()
        for line in lines:
            if line.count(player) == 2 and line.count(' ') == 1:
                score += 10
            elif line.count(self.getOpponent(player)) == 2 and line.count(' ') == 1:
                score -= 10

        return score

    def get_lines(self):
        lines = []
        for i in range(3):  # Rows
            lines.append(self.board[i * 3:(i + 1) * 3])

        for i in range(3):  # Columns
            lines.append([self.board[j * 3 + i] for j in range(3)])

        lines.append([self.board[i] for i in [0, 4, 8]])  # Diagonals
        lines.append([self.board[i] for i in [2, 4, 6]])
        return lines

    def getOpponent(self, player):
        return 'X' if player == 'O' else 'O'

    def check_block_move(self, player_letter):
        letter_opponent = self.getOpponent(player_letter)
        for move in self.get_available_moves():
            self.make_move(move, letter_opponent)
            if self.winner(move, letter_opponent):
                self.undo_move(move)
                return move
            self.undo_move(move)
        return None


def minimax(game, depth, isMaxPlayer=True, isPruning=False, alpha=-math.inf, beta=math.inf, isTicTac=True):
    if depth == 0 or game.is_game_over():
        return None, game.evaluate_score()

    val_max = - float('inf')
    val_min = float('inf')
    best_move = None

    if isMaxPlayer:
        for move in game.get_available_moves():

            if isTicTac:
                game.make_move(move, 'X')  # x is maximizer
                _, val_temp = minimax(game, depth - 1, isMaxPlayer=isMaxPlayer, isPruning=isPruning)
            else:
                game.make_move(move, 1)
                _, val_temp = minimax(game, depth - 1, alpha=alpha, beta=beta, isMaxPlayer=isMaxPlayer,
                                      isTicTac=isTicTac, isPruning=isPruning)

            game.undo_move(move)

            if val_temp > val_max:
                val_max = val_temp
                best_move = move

            if isPruning:
                alpha = max(alpha, val_temp)
                if beta <= alpha:
                    break
        return best_move, val_max

    else:
        for move in game.get_available_moves():
            if isTicTac:
                game.make_move(move, 'O')  # x is minimizer
                _, val_temp = minimax(game, depth - 1, True, isPruning=isPruning)
            else:
                game.make_move(move, 2)
                _, val_temp = minimax(game, depth - 1, alpha=alpha, beta=beta, isMaxPlayer=True, isTicTac=isTicTac)

            game.undo_move(move)

            if val_temp < val_min:
                val_min = val_temp
                best_move = move

            if isPruning:
                beta = min(beta, val_temp)
                if beta <= alpha:
                    break

        return best_move, val_min


def q_learning(game, q_table, state, letter, epsilon=0.1, alpha=0.1, gamma=0.9, isConnect4=False):
    if random.random() < epsilon:
        move = random.choice(game.get_available_moves())
    else:
        move = max(q_table[state], key=q_table[state].get, default=random.choice(game.get_available_moves()))

    game.make_move(move, letter)

    if isConnect4:
        state_new = game.get_state()
    else:
        state_new = tuple(game.board)

    if state_new not in q_table: q_table[state_new] = defaultdict(float)

    if isConnect4:
        reward = game.evaluate_score_complex()  # Connect 4
    else:
        reward = game.evaluate_score_complex(letter)  # TicTacToe

    q_val_current = q_table[state][move]
    q_val_next_max = max(q_table[state_new].values(), default=0)
    q_val_next = (1 - alpha) * q_val_current + alpha * (reward + gamma * q_val_next_max)
    q_table[state][move] = q_val_next


def playTicTacToe(isPruning=False):
    game = TicTacToe()  # Initialize the game

    game.turn = 'human'  # Decide who starts

    while not game.is_game_over():
        if game.turn == 'human':
            move = int(input("Enter your move (0-8): "))
            made_move = game.make_move(move, 'O')
            if not made_move:
                continue
        else:
            print("AI's turn:")
            move, _ = minimax(game, depth=4, isMaxPlayer=True, isPruning=isPruning)
            game.make_move(move, 'X')
            print(f"AI chose position {move}")

        game.print_board()
        game.turn = 'human' if game.turn == 'ai' else 'ai'

        if game.is_game_over():
            if game.current_winner:
                winner = 'Human' if game.current_winner == 'O' else 'AI'
                print(f"{winner} wins!")
            else:
                print("It's a draw!")
            break


def playConnect4():
    game = ConnectFour()

    while not game.game_over:
        if game.turn == 0:  # Human's turn
            col = int(input("Player 1, make your selection (0-6): "))
            if game.is_valid_location(col):
                game.make_move(col, 1)  # Player 1 is 1
        else:  # AI's turn
            print("AI is making its move...")
            col, _ = minimax(game, 5, True, isTicTac=False)  # Depth set at 5
            if col is not None:
                game.make_move(col, 2)  # AI is 2

        game.print_board()
        if game.winning_move(game.turn + 1):
            print(f"Player {game.turn + 1} wins!")
            game.game_over = True

        game.turn = (game.turn + 1) % 2

        if game.is_game_over():
            print("Game Over!")
            game.game_over = True


def playAgainstDefaultOpponentConnect4(isPruning=False, isMinimax=True, playTimes=1, alpha=0.1, gamma=0.9, epsilon=0.1):
    results = {'win': 0, 'loss': 0, 'draw': 0}
    result_trend = []
    t0 = time.time()

    if isMinimax:
        checkpoints = [i for i in range(1, playTimes, 10)]

        for i in range(playTimes):
            if isMinimax:
                result = SmartConnect4().playGameMinimax(isPruning=isPruning)
                result_trend.append(result)
                if result == 'win':
                    results['win'] += 1
                elif result == 'loss':
                    results['loss'] += 1
                else:
                    results['draw'] += 1

    else:
        game = SmartConnect4()
        checkpoints = [i for i in range(1, playTimes, 100)]
        results, result_trend = game.playGameQLearning(episodes=playTimes, alpha=alpha, gamma=gamma,
                                                                  epsilon=epsilon)

    t1 = time.time()
    t = t1 - t0

    plotOverallGameTrend(result_trend)

    plotPerformanceAtCheckpoints(result_trend, checkpoints, isDefaultOpponent=True)  # plot trend at certain checkpoint

    printGameResults(results, len(result_trend))
    print(f"Total execution time={t}s")


def playAgainstDefaultOpponentTicTacToe(isPruning=False, isMinimax=True, playTimes=1):
    results = {'win': 0, 'loss': 0, 'draw': 0}
    result_trend = []
    t0 = time.time()

    if isMinimax:  # play Minimax and store result
        for i in range(playTimes):
            result = SmartTicTacToe('X').playGameMinimax(isPruning=isPruning)
            result_trend.append(result)
            if result == 'win':
                results['win'] += 1
            elif result == 'loss':
                results['loss'] += 1
            else:
                results['draw'] += 1

    else:  # play Connect 4
        t0 = time.time()
        results, result_trend = SmartTicTacToe('X').playGameQLearning(episodes=playTimes)

    t1 = time.time()
    t = t1 - t0

    plotOverallGameTrend(result_trend)  # plot overall win rate

    checkpoints = [i for i in range(1, playTimes, 100)]
    plotPerformanceAtCheckpoints(result_trend, checkpoints, isDefaultOpponent=True)  # plot trend at certain checkpoint

    printGameResults(results, len(result_trend))
    print(f"Total execution time={t}s")


def plotGameTrend(result_trend):
    cumulative_wins = [0]
    cumulative_losses = [0]
    cumulative_draws = [0]

    for ret in result_trend:
        cumulative_wins.append(cumulative_wins[-1] + (ret == 'win'))
        cumulative_losses.append(cumulative_losses[-1] + (ret == 'loss'))
        cumulative_draws.append(cumulative_draws[-1] + (ret == 'draw'))

    # cumulative_wins = cumulative_wins[1:]
    # cumulative_losses = cumulative_losses[1:]
    # cumulative_draws = cumulative_draws[1:]

    # Setting up the plot
    plt.figure(figsize=(10, 6))
    plt.title('Agent Outcomes Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Outcomes')

    episodes = list(range(len(result_trend) + 1))

    plt.plot(episodes, cumulative_wins, label='Wins', color='green')
    plt.plot(episodes, cumulative_losses, label='Losses', color='red')
    plt.plot(episodes, cumulative_draws, label='Draws', color='blue')

    # Adding a legend to explain which line is which
    plt.legend()
    plt.show()


def plotOverallGameTrend(result_trend, window_size=50):
    wins = [outcome == 'win' for outcome in result_trend]
    losses = [outcome == 'loss' for outcome in result_trend]
    draws = [outcome == 'draw' for outcome in result_trend]

    cumulative_wins = np.cumsum(wins)
    cumulative_losses = np.cumsum(losses)
    cumulative_draws = np.cumsum(draws)

    len_result = len(result_trend) + 1
    win_rates = cumulative_wins / np.arange(1, len_result)
    loss_rates = cumulative_losses / np.arange(1, len_result)
    draw_rates = cumulative_draws / np.arange(1, len_result)

    plt.figure(figsize=(14, 7))
    plt.plot(win_rates, label='Win Rate', color='green')
    plt.plot(loss_rates, label='Loss Rate', color='red')
    plt.plot(draw_rates, label='Draw Rate', color='blue')

    plt.title('Win, Loss, and Draw Rates Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Rate')
    plt.legend()
    plt.show()


# check points e.g.100, 500, 1000
def plotPerformanceAtCheckpoints(results, checkpoints, isDefaultOpponent=False):
    q_learning_wins = [results[:i].count('win') / i for i in checkpoints]
    minimax_wins = [results[:i].count('loss') / i for i in checkpoints]
    draw = [results[:i].count('draw') / i for i in checkpoints]

    plt.figure(figsize=(10, 5))

    if not isDefaultOpponent:
        plt.plot(checkpoints, q_learning_wins, label='Q-learning Win Rate', marker='o')
        plt.plot(checkpoints, minimax_wins, label='Minimax Win Rate', marker='s')
        plt.plot(checkpoints, draw, label='Draw Rate', marker='d')
    else:
        plt.plot(checkpoints, q_learning_wins, label='Win Rate', marker='o')
        plt.plot(checkpoints, minimax_wins, label='Loss Rate', marker='s')
        plt.plot(checkpoints, draw, label='Draw Rate', marker='d')

    plt.title('Performance Over Time')
    plt.xlabel('Number of Games Played')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True)
    plt.show()


def MinimaxAgainstQlearningPlayingConnect4(playTimes=100):
    game = SmartConnect4()
    t0 = time.time()

    results, result_trend = game.playGameC4MinimaxAgainstQlearning(isPruning=True, episodes=playTimes)
    t1 = time.time()
    dt = t1 - t0

    plotOverallGameTrend(result_trend)

    checkpoints = [10, 50, 100, 250, 500, 750, 1000]
    plotPerformanceAtCheckpoints(result_trend, checkpoints)

    len_game = len(result_trend)
    for k, v in results.items():
        if k == 'win':
            print(f'Q-Learning wins {v} times out of {len_game} games')
        elif k == 'loss':
            print(f'Minimax wins {v} times out of {len_game} games')
        else:
            print(f'Draw {v} times out of {len_game} games')

    print(f'Total play time: {dt}s')


def MinimaxAgainstQlearningPlayingTicTacToe(playTimes=100):
    game = SmartTicTacToe('X')  # X for Qlearning, O for Minimax
    t0 = time.time()
    results, result_trend = game.playTicTacToeMinimaxAgainstQlearning(isPruning=True, episodes=playTimes)
    t1 = time.time()
    dt = t1 - t0

    plotOverallGameTrend(result_trend)

    checkpoints = [10, 100, 250, 500, 750, 1000, 2000, 4000, 5000]
    plotPerformanceAtCheckpoints(result_trend, checkpoints)

    printGameResults(results, len(result_trend))
    print(f'Total play time: {dt}')

    # len_game =
    # for k, v in results.items():
    #     if k == 'win':
    #         print(f'Q-Learning wins {v} times out of {len_game} games')
    #     elif k == 'loss':
    #         print(f'Minimax wins {v} times out of {len_game} games')
    #     else:
    #         print(f'Draw {v} times out of {len_game} games')


def printGameResults(results, game_duration):  # results: dictionary, game_duration: list
    for k, v in results.items():
        if k == 'win':
            print(f'\nQ-Learning wins {v} times out of {game_duration} games')
        elif k == 'loss':
            print(f'Minimax wins {v} times out of {game_duration} games')
        else:
            print(f'Draw {v} times out of {game_duration} games')


def main():
    # playTicTacToe(isPruning=True)
    # playConnect4()

    # Q4.1 & Q4.3
    playAgainstDefaultOpponentTicTacToe(isPruning=True, isMinimax=True, playTimes=1000)  # Minimax
    # playAgainstDefaultOpponentTicTacToe(isMinimax=False, playTimes=50000)  # Q learning

    # Q4.2 & Q4.3
    # playAgainstDefaultOpponentConnect4(isPruning=True, isMinimax=True, playTimes=200)  # Minimax
    # playAgainstDefaultOpponentConnect4(isMinimax=False, playTimes=10000)  # Q learning

    # Q4.4 & Q4.6
    # MinimaxAgainstQlearningPlayingTicTacToe(playTimes=1000)

    # Q4.5 & Q4.6
    # MinimaxAgainstQlearningPlayingConnect4(playTimes=1000)


if __name__ == '__main__':
    main()
