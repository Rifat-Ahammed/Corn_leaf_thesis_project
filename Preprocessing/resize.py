from PIL import Image, ImageOps
import os

folder_dir = r"C:\Users\Rifat\OneDrive\Documents\Thesis project\raw_data\Blight"
output_dir = r"C:\Users\Rifat\OneDrive\Documents\Thesis project\data_resized\Blight"

for images in os.listdir(folder_dir):
    if (images.endswith(".png") or images.endswith(".jpg") or
        images.endswith(".jpeg") or images.endswith(".PNG") or
        images.endswith(".JPEG") or images.endswith(".JPG")):
        
        name, _ = os.path.splitext(images)
        print(name)

        path = os.path.join(folder_dir, images)
        image = Image.open(path)

        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Calculate the aspect ratio and resize accordingly
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:
            # Image is wider than it is tall, resize based on width
            new_width = 768
            new_height = int(768 / aspect_ratio)
        else:
            # Image is taller than it is wide, resize based on height
            new_height = 768
            new_width = int(768 * aspect_ratio)

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Create a new image with a transparent background (768x768)
        new_image = Image.new('RGB', (768, 768), (192, 192, 192))

        # Paste the resized image onto the new image (centered)
        paste_position = ((768 - new_width) // 2, (768 - new_height) // 2)
        new_image.paste(resized_image, paste_position)

        # Convert to RGB before saving as JPEG if not already in RGB mode
        if new_image.mode != 'RGB':
            new_image = new_image.convert('RGB')

        # Save the new image as JPEG
        new_image.save(os.path.join(output_dir, name + '.jpg'))
