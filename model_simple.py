import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4Model(nn.Module):
	def __init__(self, device):
		super(Connect4Model, self).__init__()

		self.device = device

		# Define the layers that will be used

		# start with a convolutional layer
		self.initial_conv = nn.Conv2d(3, 128, 3, stride=1, padding=1)
		self.initial_bn = nn.BatchNorm2d(128)

		# Res blocks
		# RES BLOCK 1
		self.res1_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.res1_bn1 = nn.BatchNorm2d(128)
		self.res1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.res1_bn2 = nn.BatchNorm2d(128)

		# RES BLOCK 2
		self.res2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.res2_bn1 = nn.BatchNorm2d(128)
		self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
		self.res2_bn2 = nn.BatchNorm2d(128)

		# value head
		# conv, fc, fc, tanh
		self.value_conv = nn.Conv2d(128, 3, kernel_size=1) # value head
		self.value_bn = nn.BatchNorm2d(3)
		self.value_fc1 = nn.Linear(3*6*7, 32)
		self.value_fc2 = nn.Linear(32, 1)

		# policy head
		# conv, fc, logsoftmax
		self.policy_conv = nn.Conv2d(128, 32, kernel_size=1) # policy head
		self.policy_bn1 = nn.BatchNorm2d(32)
		self.policy_fc = nn.Linear(6*7*32, 7)
		self.policy_ls = nn.LogSoftmax(dim=1)

		self.to(device)


	def forward(self,x):
		# Define how the layers connect

		# start with a convolutional layer
		x = x.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
		x = F.relu(self.initial_bn(self.initial_conv(x)))

		# res blocks
		# RES BLOCK 1
		res = x # make a copy to skip convolutions
		x = F.relu(self.res1_bn1(self.res1_conv1(x)))
		x = self.res1_bn2(self.res1_conv2(x))
		x += res # add residual
		x = F.relu(x)

		# RES BLOCK 2
		res = x # make a copy to skip convolutions
		x = F.relu(self.res2_bn1(self.res2_conv1(x)))
		x = self.res2_bn2(self.res2_conv2(x))
		x += res # add residual
		x = F.relu(x)

		# value head
		v = F.relu(self.value_bn(self.value_conv(x))) # value head
		v = v.view(-1, 3*6*7)  # batch_size X channel X height X width
		v = F.relu(self.value_fc1(v))
		value = torch.tanh(self.value_fc2(v))

		# policy head
		p = F.relu(self.policy_bn1(self.policy_conv(x))) # policy head
		p = p.view(-1, 6*7*32)
		p = self.policy_fc(p)
		policy = self.policy_ls(p).exp()

		return policy, value

	def predict(self, board):
		board = self.game.board_to_one_hot(board)
		board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
		board = board.view(1, 3, 6, 7) # 1 observation, 3 channels, 6 board rows, 7 board columns
		self.eval()
		with torch.no_grad():
			policy, value = self.forward(board)

		# return action and value copied from computation graph
		return policy.data.cpu().numpy()[0], value.data.cpu().numpy()[0]

if __name__ == "__main__":
	from torchinfo import summary

	if torch.cuda.is_available():
		device = torch.device('cuda')

	model = Connect4Model(device) # instantiate model
	architecture_summary = summary(model, input_size=(16,3,6,7),verbose=0)

	print(architecture_summary)

