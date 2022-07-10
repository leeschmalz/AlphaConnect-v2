import torch

from game import Connect4Game
from model import Connect4Model
from trainer import Trainer

device = ('cuda' if torch.cuda.is_available() else 'cpu')


args = {
    'batch_size': 64,
    'numIters': 500,                                # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 100,                                  # Number of full games (episodes) to run during each iteration
    'epochs': 2,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth',                # location to save latest set of weights
    'device':device,
    'learning_rate':5e-4,
    'batch_with_replacement':False,
    'temperature':5,
    'verbose':1                 
}

game = Connect4Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = Connect4Model(board_size, action_size, torch.device(device)) # instantiate model

trainer = Trainer(game, model, args)
trainer.learn()
