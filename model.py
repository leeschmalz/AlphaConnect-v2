import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4Model(nn.Module):

    def __init__(self, board_size, action_size, device):

        super(Connect4Model, self).__init__()

        self.device = device
        self.size = board_size[0]*board_size[1] # ex 6x7 board = 42 neurons
        self.action_size = action_size

        self.hidden_layer_neurons = 1024

        self.fc1 = nn.Linear(in_features=self.size, out_features=self.hidden_layer_neurons)
        self.fc2 = nn.Linear(in_features=self.hidden_layer_neurons, out_features=self.hidden_layer_neurons)
        self.fc3 = nn.Linear(in_features=self.hidden_layer_neurons, out_features=self.hidden_layer_neurons)
        self.fc4 = nn.Linear(in_features=self.hidden_layer_neurons, out_features=self.hidden_layer_neurons)
        self.fc5 = nn.Linear(in_features=self.hidden_layer_neurons, out_features=self.hidden_layer_neurons)
        self.fc6 = nn.Linear(in_features=self.hidden_layer_neurons, out_features=self.hidden_layer_neurons)
        self.fc7 = nn.Linear(in_features=self.hidden_layer_neurons, out_features=self.hidden_layer_neurons)
        self.fc8 = nn.Linear(in_features=self.hidden_layer_neurons, out_features=self.hidden_layer_neurons)

        # Two heads on our network
        self.action_head = nn.Linear(in_features=self.hidden_layer_neurons, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=self.hidden_layer_neurons, out_features=1)

        self.to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = F.relu(x)

        x = self.fc6(x)
        x = F.relu(x)

        x = self.fc7(x)
        x = F.relu(x)

        x = self.fc8(x)
        x = F.relu(x)

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        return F.softmax(action_logits, dim=1), torch.tanh(value_logit) # return action, value

    def predict(self, board):
        board = board.flatten()

        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.size)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0] # return action and value copied from computation graph
