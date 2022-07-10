import numpy as np


class Connect4Game:
    """
    standard Connect 4:
    7 columns x 6 rows
    win condition: 4 in a row vertically, horizontally, or diagonally.
    """

    def __init__(self):
        self.columns = 7
        self.rows = 6

    def get_init_board(self):
        b = np.zeros((self.rows,self.columns), dtype=np.int)
        return b

    def get_board_size(self):
        return (self.rows, self.columns)

    def get_action_size(self):
        return self.columns

    def place_piece(self,board,action,player):
        # count the number of 0's in the column, play above the last non-zero piece
        b = np.copy(board)
        b[sum(b[:,action] == 0)-1,action] = player
        
        return b

    def get_next_state(self, board, player, action):
        b = np.copy(board)
        b = self.place_piece(b,action,player)

        # Return the new game, but
        # change the perspective of the game with negative
        return (b, -player)

    def has_legal_moves(self, board):
        # check if there are any legal moves remaining
        if sum(board.flatten()==0) > 0:
            return True
        else:
            return False

    def get_valid_moves(self, board):
        # All moves are invalid by default
        valid_moves = [0] * self.columns

        for index in range(self.columns):
            # check each column, if its not full -> valid
            if sum(board[:,index] == 0) > 0:
                valid_moves[index] = 1

        return valid_moves

    def is_win(self, board, player):
        '''
        return True if player has 4 in a row, else False.
        '''
        # horizontal
        winner = 0
        done = False
        for i in range(6):
            for j in range(4):
                if all(board[i,j:j+4] == player):
                    return True
                
        # vertical
        for j in range(7):
            for i in range(3):
                if all(board[i:i+4,j] == 1):
                    return True
        
        # diagonal top left to bottom right
        for row in range(3):
            for col in range(4):
                if board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] == board[row + 3][col + 3] == player:
                    return True
                
        # diagonal bottom left to top right
        for row in range(5, 2, -1):
            for col in range(3):
                if board[row][col] == board[row - 1][col + 1] == board[row - 2][col + 2] == board[row - 3][col + 3] == player:
                    return True

        return False

    def get_reward_for_player(self, board, player):
        # return None if not ended, 1 if player 1 wins, -1 if player 1 lost, 0 if draw
        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if self.has_legal_moves(board):
            return None

        return 0

    def get_canonical_board(self, board, player):
        # if the board is from player -1's perspective, flip it.
        return player * board

    def invert_board(self, board):
        return board*-1

