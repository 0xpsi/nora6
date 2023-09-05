import torch
import torch.nn.functional as F
import torch.nn as nn
#"""
class PlanetNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
		self.fc1 = nn.Linear(16 * 22 * 22, 2048)
		self.fc2 = nn.Linear(2048, 256)
		self.fc3 = nn.Linear(256, 8)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)

"""
class PlanetNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(10000, 5000)
		self.fc2 = nn.Linear(5000, 2000)
		self.fc3 = nn.Linear(2000, 500)
		self.fc4 = nn.Linear(500, 250)
		self.fc5 = nn.Linear(250, 100)
		self.fc6 = nn.Linear(100, 50)
		self.fc7 = nn.Linear(50, 8)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = F.relu(self.fc6(x))
		x = self.fc7(x)
		return F.log_softmax(x, dim=1)
"""
