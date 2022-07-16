import numpy as np
from game import Connect4Game

board = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, -1, 1, -1, 0, 0]])



# convert each element of board to a one-hot vector
def board_to_one_hot(board):
    board_one_hot = np.zeros((3, 6, 7))

    for i in range(6):
        for j in range(7):
            if board[i][j] == 1:
                board_one_hot[0][i][j] = 1
            elif board[i][j] == -1:
                board_one_hot[2][i][j] = 1
            else:
                board_one_hot[1][i][j] = 1

    return board_one_hot
