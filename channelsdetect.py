import numpy as np
import cv2
from PIL import Image
import os

def detect_channels(image_path):
    ext = os.path.splitext(image_path)[-1].lower()

    # if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
    #     img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #     if img is None:
    #         raise ValueError(f"Failed to read image: {image_path}")
    #     shape = img.shape
    #     if len(shape) == 2:
    #         return 1  # Grayscale
    #     else:
    #         return shape[2]  # Channels

    if not image_path.endswith(".npy"):
        raise ValueError("Please use .npy images for detection.")

    arr = np.load(image_path)

    if arr.ndim == 2:
        return 1
    elif arr.ndim == 3:
        return arr.shape[2]
    else:
        raise ValueError("Unknown image format")

# Example usage:
channels = detect_channels("converted_images_npy\HLS\pexels-3170155-9421350.npy")
print(f"Number of channels: {channels}")