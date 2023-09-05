import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import sys

import PlanetNet as n6

device = "cpu"
if torch.cuda.is_available():
	device = "cuda"

num_threads = 12

torch.set_num_threads(num_threads)

print("Device type: ", device)
print("Num devices: ", torch.cuda.device_count())
print("Num threads: ", num_threads)

input_path = "inputs/mercury.jpg"

if len(sys.argv) == 2:
	input_path = "inputs/" + sys.argv[1]

input_image = Image.open(input_path)

classes = ['earth','jupiter','mars','mercury','neptune','saturn','uranus','venus']

xform = transforms.Compose([
	transforms.Resize((100,100)),
	transforms.ToTensor(),
	transforms.Grayscale()
])
input_tensor = xform(input_image).to(device)

model_file = "nora6_model.pt"

model = n6.PlanetNet()
#model = model.to(device)
model = nn.DataParallel(model).to(device)
#checkpoint = torch.load(model_file)
model.load_state_dict(torch.load(model_file))
model.eval()

with torch.no_grad():
	print("model input tensor size: ", input_tensor.size())
	output = model(input_tensor.view(-1, 1, 100, 100))
	print("model output tensor size: ", output.size())
	print("model output:", output)
	o = torch.argmax(output)
	print("argmax of output:")
	print("\t type: ",type(o))
	print("\t shape: ", o.shape)
	print("\t val: ",o)
	x = o.clone().detach().item()

print("Type: ", type(x))
print("Output: ", classes[x])
