import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4Model_old(nn.Module):

    def __init__(self, board_size, action_size, device):

        super(Connect4Model, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, padding=0,stride=1)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, padding=0,stride=1)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, padding=0,stride=1)

        self.device = device
        self.size = board_size
        self.action_size = action_size

        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.bn1 = nn.BatchNorm2d(128)

        self.action_head = nn.Linear(in_features=64, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=64, out_features=1)

        self.to(device)

    def forward(self, x):
        x = self.conv1(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # flatten before dense layer
        x = F.relu(self.fc1(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        return F.softmax(action_logits, dim=1), torch.tanh(value_logit) # return action, value

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, 1, self.size[0], self.size[1])
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0] # return action and value copied from computation graph

class Connect4Model(nn.Module):
    def __init__(self, board_size, action_size, device, game):
        super(Connect4Model, self).__init__()

        self.game = game
        self.device = device
        self.size = board_size
        self.action_size = action_size

        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()

        self.action_head = nn.Linear(in_features=64, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=64, out_features=1)
        
        self.to(device)
        
    def forward(self,s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s

    def predict(self, board):
        board = self.game.board_to_one_hot(board)
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, 3, self.size[0], self.size[1])
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0] # return action and value copied from computation graph

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*6*7, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6*7*32, 7)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 3*6*7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 6*7*32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v