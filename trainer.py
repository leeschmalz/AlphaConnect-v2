import os
from tabnanny import verbose
import numpy as np
from random import shuffle
import time
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import joblib
from joblib import Parallel, delayed
from monte_carlo_tree_search import MCTS

class Trainer:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)

    def execute_episode(self):
        '''
        executes a single game of connect 4
        '''

        train_examples = []
        current_player = 1
        state = self.game.get_init_board()

        while True:
            # canonical board is the board from a single perspective,
            # the network makes decisions only from view of player 1

            canonical_board = self.game.get_canonical_board(state, current_player)

            # make action decision
           
            win_possible_player, winning_move = self.game.win_possible(canonical_board)
            win_possible_opponent, blocking_move = self.game.win_possible(self.game.invert_board(canonical_board))
            if win_possible_player and self.args['force_win']: # if win is available, take it
                action_probs = np.zeros(self.game.get_action_size())
                action_probs[winning_move] = 1.0
                action = winning_move
            elif win_possible_opponent and self.args['force_block_win']: # if opponent can win, block it
                action_probs = np.zeros(self.game.get_action_size())
                action_probs[blocking_move] = 1.0
                action = blocking_move
            else: # else, use MCTS to make a move
                # initialize a new tree with the current state of the game and model
                self.mcts = MCTS(self.game, self.model, self.args)

                # simulate the decision of player 1 many times
                # always player 1 since were using a canonical board
                root = self.mcts.run(self.model, canonical_board, to_play=1)

                # use the visit counts of the children of the decision node to get probs
                action_probs = [0 for _ in range(self.game.get_action_size())]
                for k, v in root.children.items():
                    action_probs[k] = v.visit_count

                # normalize so they sum to 1
                action_probs = action_probs / np.sum(action_probs)

                action = root.select_action(temperature=self.args['temperature'])

            # add to training batch
            train_examples.append((canonical_board, current_player, action_probs))

            if self.args['verbose'] > 0:
                print('\n')
                print('winning move: ', winning_move)
                print('blocking move: ', blocking_move)
                print('action_probs: ', action_probs)
                print('action: ', action)
                print(state)

            # select action based on simulations
            state, current_player = self.game.get_next_state(state, current_player, action)
            reward = self.game.get_reward_for_player(state, current_player)
            # reward is 1 if current_player wins, -1 if current_player lost
                
                
            # if the game is over
            if reward is not None:
                if self.args['verbose'] > 0:
                    print('\n')
                    print(state)
                    print('\n')
                    print(f'Player {current_player*reward} wins.')
                    print('\n')
                    print('\n')
                    
                ret = [] 
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    # [
                    # hist_state, -> the state of the board at the time of the action probs
                    # hist_action_probs, -> the action probs at the time of hist_state used to make a decision
                    # Reward, -> the reward for the player that played that move (-1 if loser, 1 if winner)
                    # ]
                    
                    # invert the reward if hist_current_player is not current_player
                    if (hist_current_player != current_player):
                        hist_reward = -reward
                    else:
                        hist_reward = reward

                    # one hot encode each position to channels first representation (3,6,7)
                    hist_state = self.game.board_to_one_hot(hist_state)

                    ret.append((hist_state, hist_action_probs, hist_reward))

                return ret

    def learn(self):
        iter = self.args['start_iter']
        while True:
            iter += 1
            print(f"Iteration: {iter}")

            train_examples = []
            
            print('generating training data...\n')
            if self.args['parallelize']:
                num_cpus = 6
                iteration_train_examples = Parallel(n_jobs=num_cpus,verbose=100)(delayed(self.execute_episode)() for eps in range(self.args['numEps']))
                for ex in iteration_train_examples:
                    train_examples.extend(ex)
            else:
                for eps in tqdm(range(self.args['numEps'])):
                    iteration_train_examples = self.execute_episode()
                    train_examples.extend(iteration_train_examples) 
            
            shuffle(train_examples)
            self.train(train_examples)
            
            filename = str(iter) + "_model.pth"
            if iter % self.args['save_freq'] == 0:
                self.save_checkpoint(folder="./"+self.args['model_dir'], filename=filename)

            random_benchmark = self.game.random_benchmark(self.model,test_games=self.args['benchmark_games'])
            greedy_benchmark = self.game.greedy_benchmark(self.model,test_games=self.args['benchmark_games'],verbose=True)

            print(f"Random benchmark: {random_benchmark}")
            print(f"Greedy benchmark: {greedy_benchmark}")


    def train(self, examples):
        '''
        inputs train examples for an episode of shape (hist_state, hist_action_probs, reward)
        calculate total loss from value and action loss
        train network
        '''

        optimizer = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'])
        pi_losses = [] # action prediction losses
        v_losses = [] # value losses

        print(f"\n Training on {len(examples)} examples. \n")
        for epoch in range(self.args['epochs']):
            print(f"\n Epoch {epoch+1} / {self.args['epochs']}\n")
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                # randomly batch train examples
                if self.args['batch_with_replacement']:
                    sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                else:
                    sample_ids = list(range(batch_idx*self.args['batch_size'],(batch_idx+1)*self.args['batch_size']) )
                    
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids if i < len(examples)]))

                # convert to tensor
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # make contiguous copies if cuda
                boards = boards.contiguous().cuda() 
                target_pis = target_pis.contiguous().cuda() 
                target_vs = target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)

                # total loss is sum of value and action losses
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print('\n')

    def loss_pi(self, targets, outputs):
        # categorical cross-entropy on action predictions
        # relu + epsilon output to prevent log(0)
        loss = -(targets * torch.log(F.relu(outputs)+1e-6)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        # mean squared error loss for value prediction
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)

        torch.save(self.model.state_dict(), filepath)
