import cv2
import numpy as np
from PIL import Image

def downscale_image_by(image, max_size,x=64):
    try:
        image = np.array(image)
        height, width = image.shape[:2]
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        new_width = (new_width // x) * x
        new_height = (new_height // x) * x
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Error downscaling image: {e}")
        return None