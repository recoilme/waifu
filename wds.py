import os
import json
from pathlib import Path
from PIL import Image
import io

def create_dataset(image_text_dir, output_file, resample=False):
    dataset = {}
    image_folder = Path(image_text_dir)
    
    # Объединяем поиск изображений в один список
    image_extensions = ('*.jpeg', '*.jpg', '*.png')
    all_images = []
    for ext in image_extensions:
        all_images.extend(image_folder.glob(ext))
    all_images = sorted(all_images)
    num_images = len(all_images)
    print(f"Found {num_images} images")

    # Константы для фильтрации
    MIN_IMAGE_SIZE = 128
    MAX_IMAGE_SIZE = 4096
    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 10000
    BLOCK_SIZE = 64

    processed_files = set()
    
    for image_path in all_images:
        if str(image_path) in processed_files:
            print(f"Warning: {image_path} was already processed!")
            continue

        txt_file = image_path.with_suffix('.txt')
        if not txt_file.exists():
            print(f"Skipping {image_path}: no text file")
            os.remove(image_path)
            continue

        try:
            # Чтение текста
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            # Проверка длины текста
            if not (MIN_TEXT_LENGTH <= len(text) <= MAX_TEXT_LENGTH):
                print(f"Skipping {image_path}: text length out of bounds")
                os.remove(image_path)
                os.remove(txt_file)
                continue

            # Обработка изображения
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                width, height = img.size

                # Проверка размеров изображения и даунсемплинг при необходимости
                if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
                    scale_factor = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    width, height = new_width, new_height
                    new_width = (width // BLOCK_SIZE) * BLOCK_SIZE
                    new_height = (height // BLOCK_SIZE) * BLOCK_SIZE
                    if new_width != width or new_height != height:
                        img = img.crop((0, 0, new_width, new_height))
                    # Сохранение изображения
                    img.save(image_path, 'JPEG', quality=97)
                    width, height = img.size
                else:
                    if resample:
                        # Обрезаем до размеров, кратных BLOCK_SIZE
                        new_width = (width // BLOCK_SIZE) * BLOCK_SIZE
                        new_height = (height // BLOCK_SIZE) * BLOCK_SIZE
                        if new_width != width or new_height != height:
                            img = img.crop((0, 0, new_width, new_height))
                            img.save(image_path, 'JPEG', quality=97)

                # Проверка минимального размера
                if new_width < MIN_IMAGE_SIZE or new_height < MIN_IMAGE_SIZE:
                    print(f"Skipping: size too small", new_width, new_height)
                    os.remove(image_path)
                    os.remove(txt_file)
                    continue

                # Создание записи для датасета
                sample_id = image_path.stem
                entry = {
                    'id': sample_id,
                    'width': new_width,
                    'height': new_height,
                    'image_path': str(image_path),
                    'text': text
                }

                # Добавляем запись в датасет, группируя по размеру
                size_key = f"{new_width}x{new_height}"
                if size_key not in dataset:
                    dataset[size_key] = []
                dataset[size_key].append(entry)

                processed_files.add(str(image_path))
                if len(processed_files) % 10000 == 0:
                    print(len(processed_files))

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    # Финальное сохранение датасета в JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Выведем статистику
    print("\nProcessing statistics:")
    print(f"Total images found: {len(all_images)}")
    print(f"Total images processed: {len(processed_files)}")
    print("Images per size group:")
    for size_key, samples in dataset.items():
        print(f"- {size_key}: {len(samples)} images")

if __name__ == "__main__":
    print('start')
    image_text_directory = "ds"
    output_json = "index.json"
    create_dataset(image_text_directory, output_json, resample)
    print('end')