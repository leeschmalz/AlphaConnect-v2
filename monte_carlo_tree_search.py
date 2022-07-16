import torch
import math
import numpy as np
from tqdm import tqdm

def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.

        temperature is a num from 0 to inf that defines the likelihood we select the action with the
        highest visit count. Where 0 is argmax of visit counts and inf is a random action.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # define the distribution of actions we sample from based on the temperature and visit counts
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


class MCTS:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

    def run(self, model, state, to_play):
        '''
        Makes a single decision in the episode through monte carlo simulation.

        returns the root of the tree
        '''
        # the root node is the board at the current state of the game for which 
        # we need to decide an action

        root = Node(0, to_play)

        # EXPAND

        # predict on root -- get value of children
        action_probs, value = model.predict(state)

        # if any children are invalid, mask action prob to 0
        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves
        action_probs /= np.sum(action_probs)

        # create children nodes and invert to_play for the next players turn
        root.expand(state, to_play, action_probs)

        for _ in range(self.args['num_simulations']):
            # num_simulations is the number of times to simulate a single decision
            # start with the root node
            node = root
            search_path = [node]

            # keep selecting the next action until we reach a leaf
            while node.expanded(): # a node that has children is expanded
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2] # the second to last node on the search path is the parent of the leaf
            state = parent.state # use the parent state to generate the next_state

            # Now we're at a leaf node and we would like to expand -- get the state at the leaf
            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            # every state is inverted from it's parent to alternate turns
            next_state = self.game.invert_board(next_state)

            # the value from the perspective of the newly inverted board, or the other player
            value = self.game.get_reward_for_player(next_state, player=1)

            # If the game has not ended
            if value is None:
                # EXPAND
                action_probs, value = model.predict(next_state)

                # mask invalid moves
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  
                action_probs /= np.sum(action_probs)

                node.expand(next_state, parent.to_play * -1, action_probs)

            # if the game is over value = reward, else use predicted value from model
            self.backpropagate(search_path, value, parent.to_play * -1)

        return root

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root. adding value for player and subtracting for -player
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1