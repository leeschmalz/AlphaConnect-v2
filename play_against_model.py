import numpy as np
import pygame
import sys
import math
import random
from model import Connect4Model
import torch
from game import Connect4Game
import time

#model to play against
model_dir = 'C:\\Users\\leesc\\Documents\\GitHub\\AlphaConnect-v2\\models_2022-07-14-22-52-07\\'
model_name = '202' + '_model.pth'
device = torch.device('cuda')

BLUE = (52, 186, 235)
GREY = (70, 71, 70)
WHITE = (255,255,255)
YELLOW = (230,230,20)

ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
    board = np.zeros((ROW_COUNT,COLUMN_COUNT))
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

	# Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

	# Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

	# Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, GREY, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, WHITE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
	
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == -1:
                pygame.draw.circle(screen, BLUE, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()

game = Connect4Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = Connect4Model(board_size, action_size, device, game)
model.load_state_dict(torch.load(f'{model_dir}{model_name}'))

board = create_board()
game_over = False
turn = random.choice([-1,1])
pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("arial", 65)

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
            
        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, WHITE, (0,0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == 1:
                pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
            else:
                pygame.draw.circle(screen, BLUE, (posx, int(SQUARESIZE/2)), RADIUS)
        pygame.display.update()
        
        if event.type == pygame.MOUSEBUTTONDOWN or turn==-1:
            pygame.draw.rect(screen, WHITE, (0,0, width, SQUARESIZE))
			#print(event.pos)
			# Ask for Player 1 Input
            if turn == 1:
                posx = event.pos[0]
                col = int(math.floor(posx/SQUARESIZE))
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, 1)
                    
                    if winning_move(board, 1):
                        label = myfont.render("You win.", 1, YELLOW)
                        screen.blit(label, (40,10))
                        game_over = True
                        if not any(board.flatten() == 0):
                            game_over = True


			# # Ask for Player 2 Input
            else:
                # predict on this
                print(np.flip(board, 0))

                action_probs, _ = model.predict(game.invert_board(np.flip(board, 0)))

                # if any children are invalid, mask action prob to 0
                valid_moves = game.get_valid_moves(game.invert_board(np.flip(board, 0)))
                action_probs = action_probs * valid_moves
                action_probs /= np.sum(action_probs)

                col = np.argmax(action_probs).item()
                time.sleep(1)
                print('\n')
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, -1)
                    
                    if winning_move(board, -1):
                        label = myfont.render("You lose.", 1, BLUE)
                        screen.blit(label, (40,10))
                        game_over = True
                        if not any(board.flatten() == 0):
                            game_over = True

            draw_board(board)

            turn *= -1
            
            if game_over:
                print(np.flip(board, 0))
                pygame.time.wait(3000)