import numpy as np
import random
import pygame
import sys

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
        for i in range(6):
            for j in range(4):
                if all(board[i,j:j+4] == player):
                    return True
                
        # vertical
        for j in range(7):
            for i in range(3):
                if all(board[i:i+4,j] == player):
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
        
    def win_possible(self, board):
        # check if win is possible this looks at the board from player 1's perspective
        # horizontal
        for i in range(6):
            for j in range(4):
                segment = board[i,j:j+4]
                # check if segment has a win spot available
                if sum(segment == 1) == 3 and sum(segment == 0) == 1:
                    rel_winning_col_index = np.where(segment == 0)[0][0]

                    # check if the winning spot is in the bottom row
                    if i == 5:
                        return True, j+np.where(segment == 0)[0][0]

                    # check if the space below the win is not empty
                    if board[i+1,j+rel_winning_col_index] != 0:
                        # return True, winning action
                        return True, j+np.where(segment == 0)[0][0]

        # vertical
        for j in range(7):
            for i in range(3):
                segment = board[i:i+4,j]
                if sum(segment == 1) == 3 and segment[0] == 0:
                    # return True, winning action
                    return True, j

        # diagonal top left to bottom right
        for row in range(3):
            for col in range(4):
                segment = board[row:row+4,col:col+4].diagonal()
                # check if there is an empty space that would make a win
                if sum(segment == 1) == 3 and sum(segment == 0) == 1:
                    rel_winning_col_index = np.where(segment == 0)[0][0]
                    # check if the winning spot is the bottom row
                    if row + rel_winning_col_index == 5:
                        return True, col+rel_winning_col_index

                    # check if there is a piece below the winning spot
                    if board[row+rel_winning_col_index+1,col+rel_winning_col_index] != 0:
                        return True, col+rel_winning_col_index
        
        # diagonal bottom left to top right
        for row in range(5, 2, -1):
            for col in range(3):
                segment = np.flip(np.diag(np.fliplr(board[row-3:row+1,col:col+4])))
                # check if there is an empty space that would make a win
                if sum(segment == 1) == 3 and sum(segment == 0) == 1:
                    rel_winning_col_index = np.where(segment == 0)[0][0]
                    # check if the winning spot is the bottom row
                    if row == 5 and rel_winning_col_index == 0:
                        return True, col
                    # check if there is a piece below the winning spot
                    if board[row-rel_winning_col_index+1,col+rel_winning_col_index] != 0:
                        return True, col+rel_winning_col_index
        
        # if no wins were found
        return False, None

    def get_greedy_action(self,board):
        '''
        take available wins, elif block opponent wins, else take random action
        '''
        win_is_possible, winning_move = self.win_possible(board)
        if win_is_possible: # check if I can win
            return winning_move
        else:
            opponent_win_possible, opponent_winning_move = self.win_possible(self.invert_board(board)) # check if opponent can win
            if opponent_win_possible:
                return opponent_winning_move
            else:
                return self.get_random_valid_action(board)
    
    def check_valid_action(self,board,action):    
        # check if column is full
        if all(board[:,action] != 0):
            valid = False
        else:
            valid = True

        return valid

    def get_random_valid_action(self,board):
        valid_action = False
        while not valid_action:
            action = random.randint(0,6)
            valid_action = self.check_valid_action(board,action)
        return action

    
    def greedy_benchmark(self,model,test_games,verbose=False):
        model_wins = 0
        model_losses = 0

        for i in range(test_games):
            print_game = False
            if i % 50 == 1 and verbose:
                print_game = True

            # initialize the game
            board = self.get_init_board()

            while True:
                # player 1 plays (model)
                action_probs, _ = model.predict(board)

                # if any children are invalid, mask action prob to 0
                valid_moves = self.get_valid_moves(board)
                action_probs = action_probs * valid_moves
                action_probs /= np.sum(action_probs)

                action = np.argmax(action_probs).item()
                board, _ = self.get_next_state(board=board, player=1, action=action)
                winner = self.get_reward_for_player(board, 1)

                if print_game:
                    print('model plays:', action)
                    print(board)
                    print('winner:', winner)
                    print('\n')
                
                if winner is not None:
                    if winner == 1:
                        model_wins += 1
                    else:
                        model_losses += 1
                    break
                
                # player 2 plays (greedy agent)
                opponent_action = self.get_greedy_action(self.invert_board(board))
                board, _ = self.get_next_state(board=board, player=-1, action=opponent_action)
                winner = self.get_reward_for_player(board, 1)
                if print_game:
                    print('opponent plays:', opponent_action)
                    print(board)
                    print('winner:', winner)
                    print('\n')

                if winner is not None:
                    if winner == 1:
                        model_wins += 1
                    else:
                        model_losses += 1
                    break
        
        return model_wins / test_games
    
    def random_benchmark(self,model,test_games):
        model_wins = 0
        model_losses = 0

        for _ in range(test_games):
            # initialize the game
            board = self.get_init_board()

            while True:
                # player 1 plays (model)
                action_probs, _ = model.predict(board)

                # if any children are invalid, mask action prob to 0
                valid_moves = self.get_valid_moves(board)
                action_probs = action_probs * valid_moves
                action_probs /= np.sum(action_probs)

                action = np.argmax(action_probs).item()

                board, _ = self.get_next_state(board=board, player=1, action=action)
                winner = self.get_reward_for_player(board, 1)
                if winner is not None:
                    if winner == 1:
                        model_wins += 1
                    else:
                        model_losses += 1
                    break
                
                # player 2 plays (random agent)
                opponent_action = self.get_random_valid_action(self.invert_board(board))
                board, _ = self.get_next_state(board=board, player=-1, action=opponent_action)
                winner = self.get_reward_for_player(board, 1)
                if winner is not None:
                    if winner == 1:
                        model_wins += 1
                    else:
                        model_losses += 1
                    break
        
        return model_wins / test_games

    def board_to_one_hot(self,board):
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
    
    def draw_board(self,screen,board):
        COLUMN_COUNT = 7
        ROW_COUNT = 6
        SQUARESIZE = 100
        RADIUS = int(SQUARESIZE/2 - 5)

        BLUE = (52, 186, 235)
        BLACK = (70, 71, 70)
        WHITE = (255,255,255)
        ORANGE = (255,100,0)

        width = COLUMN_COUNT * SQUARESIZE
        height = (ROW_COUNT+1) * SQUARESIZE

        size = (width, height)
        board = np.flip(board,0)
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(screen, BLACK, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(screen, WHITE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
        
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if board[r][c] == 1:
                    pygame.draw.circle(screen, BLUE, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                elif board[r][c] == -1:
                    pygame.draw.circle(screen, ORANGE, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
        pygame.display.update()

    def render(self,board):
        pygame.init()
        screen = pygame.display.set_mode((700,700))

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                self.draw_board(screen,board)
                pygame.display.update()