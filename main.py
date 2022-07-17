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
    'batch_size': 16,                          # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 100,                                   # Number of full games (episodes) to run during each iteration
    'epochs': 10,                                   # Number of epochs of training per iteration
    'model_dir':'models_2022-07-14-22-52-07', # 'models_' + time.strftime("%Y-%m-%d-%H-%M-%S"), # 
    'start_iter':68,
    'learning_rate':5e-4,
    'batch_with_replacement':False,
    'temperature':1,
    'verbose':0,
    'save_freq':1,
    'benchmark_games':100,
    'debug_mode':False,
    'parallelize':True
}

if args['debug_mode']:
    args['model_dir'] = 'debug_models'

game = Connect4Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = Connect4Model(board_size, action_size, device, game) # instantiate model

# make model directory
if not os.path.exists(args['model_dir']):
    os.makedirs(args['model_dir'])

    # save training args
    tf = open(f"{args['model_dir']}\\args.json", "w")
    json.dump(args,tf)
    tf.close()

    # write model summary to file
    architecture_summary = summary(model, input_size=(args['batch_size'],3,6,7),verbose=0)
    tf = open(f"{args['model_dir']}\\model_summary.txt", "w",encoding="utf-8")
    tf.write(str(architecture_summary))
    tf.close()

else:
    latest_model_ind = max([int(m.split('_')[0]) for m in os.listdir(args['model_dir']) if m.endswith('.pth')])
    model.load_state_dict(torch.load(f"{args['model_dir']}\\{latest_model_ind}_model.pth"))

# begin training
trainer = Trainer(game, model, args)
trainer.learn()
