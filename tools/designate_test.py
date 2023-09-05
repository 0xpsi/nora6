import os
import shutil

classes = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

for class in classes:
	train_path = "images/train/" + class
	test_path = "images/test/" + class
	imgs = os.listdir(path)
	for i in imgs:
		
