import os
import shutil
import random

source_dir = r"C:\Users\Yarid\Desktop\archive\Van"
train_dir = os.path.join(source_dir, "Van_Train")
test_dir = os.path.join(source_dir, "Van_Test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

random.shuffle(files)

split_point = int(len(files) * 0.7)

train_data = files[:split_point]
test_data = files[split_point:]

for item in train_data:
    shutil.move(os.path.join(source_dir, item), os.path.join(train_dir, item))

for item in test_data:
    shutil.move(os.path.join(source_dir, item), os.path.join(test_dir, item))

print(f"Files split successfully: {len(train_data)} moved to Van_Train, {len(test_data)} moved to Van_Test.")
