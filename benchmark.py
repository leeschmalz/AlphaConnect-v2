from game import Connect4Game
from model import Connect4Model
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

game = Connect4Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_test_matches = 100
model_dir = 'C:\\Users\\leesc\\Documents\\GitHub\\AlphaConnect-v2\\model_2022-07-11-00-07-08\\'

# list directory of models
models = [m for m in os.listdir(model_dir) if m.endswith('.pth')]

ind = []
random_benchmark = []
greedy_benchmark = []

model_ind = 0
for model_name in tqdm(models):
    model_ind+=1
    model = Connect4Model(board_size, action_size, device)
    model.load_state_dict(torch.load(f'{model_dir}{model_name}'))

    model_wins = 0
    # benchmark with random agent
    for i in range(n_test_matches):
        # initialize the game
        board = game.get_init_board()
        reward = None

        while True:
            # player 1 plays (model)
            action_probs, _ = model.predict(board)
            action = np.argmax(action_probs).item()
            board, _ = game.get_next_state(board=board, player=1, action=action)
            winner = game.get_reward_for_player(board, 1)
            if winner is not None:
                if winner == 1:
                    model_wins += 1
                break
            
            # player -1 plays (random agent)
            opponent_action = game.get_random_valid_action(game.invert_board(board))
            board, _ = game.get_next_state(board=board, player=-1, action=opponent_action)
            winner = game.get_reward_for_player(board, 1)
            if winner is not None:
                if winner == 1:
                    model_wins += 1
                break

    random_benchmark.append(model_wins / n_test_matches)

    model_wins = 0
    # benchmark with greedy agent
    for i in range(n_test_matches):
        # initialize the game
        board = game.get_init_board()
        reward = None

        while True:
            # player 1 plays (model)
            action_probs, _ = model.predict(board)
            action = np.argmax(action_probs).item()
            board, _ = game.get_next_state(board=board, player=1, action=action)
            winner = game.get_reward_for_player(board, 1)
            if winner is not None:
                if winner == 1:
                    model_wins += 1
                break
            
            # player -1 plays (greedy agent)
            opponent_action = game.get_greedy_action(game.invert_board(board))
            board, _ = game.get_next_state(board=board, player=-1, action=opponent_action)
            winner = game.get_reward_for_player(board, 1)
            if winner is not None:
                if winner == 1:
                    model_wins += 1
                break

    greedy_benchmark.append(model_wins / n_test_matches)
    ind.append(model_ind)

# create csv with columns: model_name, random_benchmark, greedy_benchmark
benchmarks = pd.DataFrame({'i':ind,'model_name': models, 'random_benchmark': random_benchmark, 'greedy_benchmark': greedy_benchmark})
benchmarks.to_csv(model_dir + 'benchmark.csv', index=False)

# plot benchmark
import matplotlib.pyplot as plt
plt.plot(list(benchmarks['i']), list(benchmarks['random_benchmark']), label = "random_benchmark")
plt.plot(list(benchmarks['i']), list(benchmarks['greedy_benchmark']), label = "greedy_benchmark")
plt.legend()
# save plot
plt.savefig(model_dir + 'benchmark.png')
