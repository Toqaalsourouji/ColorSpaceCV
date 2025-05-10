import cv2
import numpy as np
import os

# this code converts images from various color spaces to numpy arrays and saves them in a specified directory
color_spaces = {
    
    'BGRA': cv2.COLOR_BGR2BGRA,
    'GRAY': cv2.COLOR_BGR2GRAY,
    'HLS': cv2.COLOR_BGR2HLS,
    'HSV': cv2.COLOR_BGR2HSV,
    'LAB': cv2.COLOR_BGR2LAB,
    'RGB': cv2.COLOR_BGR2RGB,
    'RGBA': cv2.COLOR_BGR2RGBA,
    'XYZ': cv2.COLOR_BGR2XYZ,
    'YUV': cv2.COLOR_BGR2YUV,
    
}

input_path = "converted_images"     
output_path = "converted_images_npy"  

os.makedirs(output_path, exist_ok=True)

for space, code in color_spaces.items():
    input_dir = os.path.join(input_path, space)
    output_dir = os.path.join(output_path, space)
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            img_path = os.path.join(input_dir, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Failed to read: {img_path}")
                continue

            if space == 'RGB':
                converted = cv2.cvtColor(img, code)
            elif space == 'GRAY':
                converted = cv2.cvtColor(img, code)
                converted = converted[..., np.newaxis]  # Add dummy channel dimension
            else:
                converted = cv2.cvtColor(img, code)

            save_name = os.path.splitext(file)[0] + ".npy"
            np.save(os.path.join(output_dir, save_name), converted)
            print(f"Saved: {space}/{save_name}")
