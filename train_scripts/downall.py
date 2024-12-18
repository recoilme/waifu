import os
import shutil
import zlib
import zipfile

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
        # Кроп изображения: обрезаем 64 пикселей снизу
        image = image[:new_height - 64, :]
        image = Image.fromarray(image)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Error downscaling image: {e}")
        return None

def process_images_in_directory(input_dir, output_dir, max_size=1600):
    # Создаем выходную директорию, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Рекурсивно обходим все директории и файлы
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            try:
                input_path = os.path.join(root, filename)
                
                # Проверяем, является ли файл изображением или ZIP-архивом
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    process_image(input_path, output_dir, max_size)
                elif filename.lower().endswith('.zip'):
                    print(f"processing {filename}")
                    process_zip(input_path, output_dir, max_size)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    print('done')

def process_image(input_path, output_dir, max_size):
    try:
        # Открываем изображение
        image = Image.open(input_path)

        # Обрезаем изображение
        image = downscale_image_by(image, max_size, 64)
        if image is None:
            print(f"Skipping {input_path} due to cropping error.")
            return

        # Генерируем уникальное имя файла с использованием CRC32
        crc32_hash = zlib.crc32(input_path.encode('utf-8')) & 0xffffffff
        output_filename = f"{crc32_hash}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Сохраняем обработанное изображение
        image.save(output_path, quality=96)
        #print(f"Saved processed image: {output_path}")

    except Exception as e:
        print(f"Error processing image {input_path}: {e}")

def process_zip(zip_path, output_dir, max_size):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Читаем изображение из архива
                    with zip_ref.open(zip_info) as img_file:
                        image = Image.open(img_file)

                        # Обрезаем изображение
                        image = downscale_image_by(image, max_size, 64)
                        if image is None:
                            print(f"Skipping {zip_info.filename} due to cropping error.")
                            continue

                        # Генерируем уникальное имя файла с использованием CRC32
                        crc32_hash = zlib.crc32(zip_info.filename.encode('utf-8')) & 0xffffffff
                        output_filename = f"{crc32_hash}.jpg"
                        output_path = os.path.join(output_dir, output_filename)

                        # Сохраняем обработанное изображение
                        image.save(output_path, quality=96)
                        #print(f"Saved processed image from zip: {output_path}")

    except Exception as e:
        print(f"Error processing zip {zip_path}: {e}")

# Пример использования
input_dir = '/Users/v.kulibaba/Downloads/nsfw'
output_dir = '/Users/v.kulibaba/Downloads/nsfw_milf2'
process_images_in_directory(input_dir, output_dir, 1280)
print('done')