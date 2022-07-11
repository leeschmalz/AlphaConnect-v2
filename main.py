import torch
import time
import os
import json
from torchinfo import summary

from game import Connect4Game
from model import Connect4Model
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'batch_size': 16,
    'numIters': int(1e9),                           # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 50,                                   # Number of full games (episodes) to run during each iteration
    'epochs': 30,                                   # Number of epochs of training per iteration
    'model_dir':'models_' + time.strftime("%Y-%m-%d-%H-%M-%S"),
    'checkpoint_path': 'model.pth',                 # location to save latest set of weights
    'learning_rate':5e-4,
    'batch_with_replacement':False,
    'temperature':0,
    'verbose':0,
    'save_freq':1,
    'benchmark_games':1000
}

game = Connect4Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = Connect4Model(board_size, action_size, device) # instantiate model

# make model directory
if not os.path.exists(args['model_dir']):
    os.makedirs(args['model_dir'])

# save training args
tf = open(f"{args['model_dir']}\\args.json", "w")
json.dump(args,tf)
tf.close()

# write model summary to file
architecture_summary = summary(model, input_size=(1,42),verbose=0)
tf = open(f"{args['model_dir']}\\model_summary.txt", "w",encoding="utf-8")
tf.write(str(architecture_summary))
tf.close()

# begin training
trainer = Trainer(game, model, args)
trainer.learn()
