import os
import random

directory = 'images-aug/planets'
length = 15  # Length of random number

def rename_files(root_dir):
    # Generate a list of all file paths in all subdirectories
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # Rename each file with a random fixed-length number
    random.seed(0)  # Set seed for reproducibility
    for file_path in file_paths:
        base, ext = os.path.splitext(file_path)
        new_name = str(random.randint(0, 10**length)).zfill(length)  # 8-digit random number
        while new_name + ext in file_paths:  # Avoid duplicates
            new_name = str(random.randint(0, 10**length)).zfill(length)
        new_name = os.path.join(os.path.dirname(file_path), new_name + ext)
        os.rename(file_path, new_name)

rename_files(directory)
