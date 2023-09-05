import torch
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

import PlanetNet as n6

epochs = 300
train_batch_size = 512
test_batch_size = 64
learning_rate = 0.0005
test_num = 1500
num_threads = 4

model_file = "nora6_model.pt"

device = "cpu"
if torch.cuda.is_available():
	device = "cuda"

# tell python to use all cpu cores
torch.set_num_threads(num_threads)

print("Using device: ", device)
print("Num devices: ", torch.cuda.device_count())
print("Num threads: ", num_threads)

model = n6.PlanetNet()
#model.to(device)
model = nn.DataParallel(model).to(device)

xform = transforms.Compose([
	# add a resize transform
	transforms.Resize((100,100)),
	transforms.ToTensor(),
	transforms.Grayscale()
])

raw_data = datasets.ImageFolder(root="images-aug/planets", transform=xform)
print("Image Classes:")
print(raw_data.find_classes("images-aug/planets")[1])

dataz = torch.utils.data.random_split(raw_data, [test_num, len(raw_data)-test_num])
raw_test = dataz[0]
raw_train = dataz[1]

# print the size of the training and test sets
print("Training set size: ", len(raw_train))
print("Test set size: ", len(raw_test))

test_loader = torch.utils.data.DataLoader(raw_test, shuffle=True, num_workers=2, batch_size=test_batch_size)
train_loader = torch.utils.data.DataLoader(raw_train, shuffle=True, num_workers=2, batch_size=train_batch_size)

epochz = []
train_loss = []
test_loss = []

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	model.train()
	loss = 0
	for batch, (X,y) in enumerate(dataloader):
		X,y = X.to(device), y.to(device)
		#print("X.shape: ", X.shape)
		#print("X.view(-1,100*100)", X.view(-1, 100*100).shape)
		#yp = model(X.view(-1, 100*100))
		yp = model(X)
		loss = loss_fn(yp, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
	train_loss.append(loss.item())


def test(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	model.eval()
	loss, correct = 0, 0
	with torch.no_grad():
		for X,y in dataloader:
			X,y = X.to(device), y.to(device)
			#yp = model(X.view(-1, 100*100))
			yp = model(X)
			loss += loss_fn(yp, y).item()
			correct += (yp.argmax(1) == y).type(torch.float).sum().item()
	loss /= size
	correct /= size
	print(f"")
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")
	test_loss.append(loss)

# get current time
start_time = time.time()

for t in range(epochs):
	print(f"Epoch {t+1} -------------------------------")
	train(train_loader, model, loss_fn, optimizer)
	test(test_loader, model, loss_fn)
	epochz.append(t)

print("Done, runtime: ", time.time() - start_time, " seconds")

# save the model state dict
torch.save(model.state_dict(), model_file)

# save the epochz and train_loss data to a csv file
with open("train_loss.csv", "w") as f:
	f.write("epoch,loss\n")
	for i in range(len(epochz)):
		f.write(f"{epochz[i]},{train_loss[i]}\n")

# save the epochz and test_loss data to a csv file
with open("test_loss.csv", "w") as f:
	f.write("epoch,loss\n")
	for i in range(len(epochz)):
		f.write(f"{epochz[i]},{test_loss[i]}\n")

# plot training and test loss
plt.plot(epochz, train_loss, label="Training Loss")
plt.plot(epochz, test_loss, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# print the number of available gpu devices to cuda torch
print("Number of GPU devices: ", torch.cuda.device_count())
