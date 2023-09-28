import os
import random
import shutil
#print(os.getcwd())
origin_dir = "Origin" 
split_data_dir ="SplitData"
train_dir = os.path.join(split_data_dir, "train")
test_dir = os.path.join(split_data_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_ratio = 0.8

def copy_files(source_dir, dest_dir, file_list):
    for file_name in file_list:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.copy(source_path, dest_path)

abnormal_dir = os.path.join(origin_dir, "Abnormal")
normal_dir = os.path.join(origin_dir, "Normal")

abnormal_images = os.listdir(abnormal_dir)
normal_images = os.listdir(normal_dir)

random.shuffle(abnormal_images)
random.shuffle(normal_images)

abnormal_train_count = int(len(abnormal_images) * train_ratio)
normal_train_count = int(len(normal_images) * train_ratio)

abnormal_train_set = abnormal_images[:abnormal_train_count]
normal_train_set = normal_images[:normal_train_count]

os.makedirs(os.path.join(train_dir, "Abnormal"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "Normal"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "Abnormal"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "Normal"), exist_ok=True)

copy_files(abnormal_dir, os.path.join(train_dir, "Abnormal"), abnormal_train_set)
copy_files(normal_dir, os.path.join(train_dir, "Normal"), normal_train_set)

abnormal_test_set = abnormal_images[abnormal_train_count:]
normal_test_set = normal_images[normal_train_count:]

copy_files(abnormal_dir, os.path.join(test_dir, "Abnormal"), abnormal_test_set)
copy_files(normal_dir, os.path.join(test_dir, "Normal"), normal_test_set)

print("Completed Split")
