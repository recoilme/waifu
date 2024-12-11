import os
from PIL import Image
from downscale import downscale_image_by
import shutil


def process_images_in_directory(input_dir, output_dir, max_size=768):
    # Создаем выходную директорию, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Перебираем все файлы в директории
    for filename in os.listdir(input_dir):
        try:
            # Проверяем, является ли файл изображением
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_path = os.path.join(input_dir, filename)
                #print(f"Processing image: {input_path}")

                # Открываем изображение
                image = Image.open(input_path)

                # Обрезаем изображение
                image = downscale_image_by(image,768,64)
                if image is None:
                    print(f"Skipping {filename} due to cropping error.")
                    continue

                # Сохраняем обработанное изображение
                name, _ = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}.jpg")
                image.save(output_path, quality=96)

                shutil.copy(os.path.join(input_dir, f"{name}.txt"), os.path.join(output_dir, f"{name}.txt"))
                #print(f"Saved processed image: {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    print('done')

# Пример использования
input_directory = "/Users/v.kulibaba/Desktop/1"  # Укажите путь к папке с изображениями
output_directory = "/Users/v.kulibaba/Desktop/2"  # Укажите путь к папке для сохранения обработанных изображений
process_images_in_directory(input_directory, output_directory)