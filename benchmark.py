from game import Connect4Game
from model import Connect4Model
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_test_matches = 100
model_dir = 'C:\\Users\\leesc\\Documents\\GitHub\\AlphaConnect-v2\\models_2022-07-14-22-52-07\\'
override = True
print('Next benchmark at: ' + str(datetime.now().replace(hour=((datetime.now().hour+1) % 24),minute=0,second=0, microsecond=0)))
while True:
    # if the current minute is 0, then we want to run the benchmark
    if datetime.now().minute == 0 and datetime.now().second == 0 or override:
        override = False
        previous_performance_path = model_dir + 'benchmark_' + str(n_test_matches) + '_match.csv'
        if os.path.exists(previous_performance_path):
            # read csv
            previous_performance = pd.read_csv(previous_performance_path)
            models_inds = [str(int(a)) for a in list(previous_performance['i'])]
            # get the models that have not been benchmarked yet
            models = [m for m in os.listdir(model_dir) if m.endswith('.pth') and m.split('_')[0] not in models_inds]
        else:
            models = [m for m in os.listdir(model_dir) if m.endswith('.pth')]
        
        if len(models) == 0:
            print('No models to benchmark')
            continue

        game = Connect4Game()
        board_size = game.get_board_size()
        action_size = game.get_action_size()

        ind = []
        random_benchmark = []
        greedy_benchmark = []
        model_names = []

        for model_name in tqdm(models):
            model_ind = int(model_name.split('_')[0])
            model = Connect4Model(board_size, action_size, device, game)
            model.load_state_dict(torch.load(f'{model_dir}{model_name}'))

            win_rate = game.random_benchmark(model, n_test_matches)
            random_benchmark.append(win_rate)
            print(f'\nRandom benchmark for model {model_ind}: {win_rate}')

            win_rate = game.greedy_benchmark(model, n_test_matches)
            greedy_benchmark.append(win_rate)
            print(f'Greedy benchmark for model {model_ind}: {win_rate}')

            model_names.append(model_name)
            ind.append(model_ind)

        # create csv with columns: model_name, random_benchmark, greedy_benchmark
        benchmarks = pd.DataFrame({'i':ind,'model_name': models, 'random_benchmark': random_benchmark, 'greedy_benchmark': greedy_benchmark})

        # stack previous performance to new benchmarks
        if os.path.exists(previous_performance_path):
            benchmarks = pd.concat([previous_performance, benchmarks])

        # sort by ind
        benchmarks = benchmarks.sort_values(by='i')
        benchmarks.to_csv(model_dir + 'benchmark_' + str(n_test_matches) + '_match.csv', index=False)

        # plot benchmark
        plt.plot(list(benchmarks['i']), list(benchmarks['random_benchmark']), label = "random_benchmark")
        plt.plot(list(benchmarks['i']), list(benchmarks['greedy_benchmark']), label = "greedy_benchmark")
        
        # x label = Training Iteration
        plt.xlabel('Training Iteration')

        # y label = Win Rate
        plt.ylabel('Win Rate')

        plt.legend()
        # save plot
        plt.savefig(model_dir + 'benchmark_' + str(n_test_matches) + '_match.png')
        plt.close()
        print('Next benchmark at: ' + str(datetime.now().replace(hour=((datetime.now().hour+1) % 24),minute=0,second=0, microsecond=0)))
