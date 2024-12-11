from PIL import Image, ImageOps
import os
import random

def augment_image(image):
    # Apply random rotation
    angle = random.choice([0, 90, 180, 270])
    image = image.rotate(angle)

    # Randomly apply horizontal flip
    if random.choice([True, False]):
        image = ImageOps.mirror(image)

    # Randomly apply vertical flip
    if random.choice([True, False]):
        image = ImageOps.flip(image)

    return image

# Directory paths
input_dir = "./data_resized/Gray_Leaf_Spot"
output_dir = "./dataset/Gray_Leaf_Spot"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each image in the input directory
for image_file in os.listdir(input_dir):
    if image_file.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path)

        augmented_image = augment_image(image)
        output_filename = f"{os.path.splitext(image_file)[0]}_aug.jpg"
        output_path = os.path.join(output_dir, output_filename)
        augmented_image.save(output_path)

        print(f"Augmented versions of {image_file} saved.")
