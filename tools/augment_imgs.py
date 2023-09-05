import os
import shutil
import random
from PIL import Image
import imghdr

source_dir = "images/planets" # directory containing original images
dest_dir = "images-aug/planets" # directory to store augmented images

TRANS_RANGE = 0.1

if not os.path.exists(dest_dir):
	os.makedirs(dest_dir)

def augment_image(src, dest):
	try:
		# Open the image
		img = Image.open(src)

		file_ext = imghdr.what(src)

		# Rotate the image by a random degree
		rotation = random.uniform(-180, 180)
		img = img.rotate(rotation, resample=Image.NEAREST)

		# Translate the image by a random offset
		width, height = img.size
		translate_x = random.uniform(-TRANS_RANGE, TRANS_RANGE) * width
		translate_y = random.uniform(-TRANS_RANGE, TRANS_RANGE) * height
		img = img.transform(img.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y))

		# remove alpha channel
		img.convert("RGB")

		# Save the augmented image
		img.save(dest, format=file_ext)
	except (OSError, UserWarning, KeyError) as e:
		print(f"Skipping file: {src}, Error: {e}")

# Iterate through all subdirectories in source_dir
for root, dirs, files in os.walk(source_dir):
	for dir in dirs:
		# Create the same subdirectory in dest_dir
		dest_subdir = os.path.join(dest_dir, dir)
		if not os.path.exists(dest_subdir):
			os.makedirs(dest_subdir)

		# Iterate through all files in the subdirectory
		for file in os.listdir(os.path.join(root, dir)):
			# Get the file path
			src_file = os.path.join(root, dir, file)

			# Create 5 augmented versions of the file
			for i in range(5):
				# Generate a unique file name for the augmented image
				dest_file = os.path.join(dest_subdir, file[:-4] + "_" + str(i) + ".jpg")

				# Perform data augmentation on the image
				augment_image(src_file, dest_file)
