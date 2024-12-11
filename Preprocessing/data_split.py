import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = "./data_resized"
output_dir = "./dataset"
train_ratio = 0.8  # 80% training data
val_ratio = 0.1   # 10% validation data
test_ratio = 0.1  # 10% test data

# Create directories for train, val, and test
for split in ['train', 'val', 'test']:
    for class_name in os.listdir(dataset_dir):
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# Split each class folder
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        images = [img for img in images if img.endswith(('.jpg', '.jpeg', '.png'))]

        # Split the data
        train_imgs, temp_imgs = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

        # Move the images to the respective folders
        for img in train_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'train', class_name, img))
        for img in val_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'val', class_name, img))
        for img in test_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'test', class_name, img))

print("Dataset split completed!")
