import ijson
from PIL import Image
from io import BytesIO
import requests
from downscale import downscale_image_by

def download_image(url):
    print(f"Downloading: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=100)  # Добавлен таймаут
        response.raise_for_status()  # Проверка на ошибки HTTP
        image = Image.open(BytesIO(response.content))
        image.load()  # Загружаем изображение, чтобы проверить его корректность
        return image
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None


def iterate_range(file_path, start, end):
    with open(file_path, 'r', encoding='utf-8') as file:
        parser = ijson.items(file, 'item')
        for i, item in enumerate(parser, start=1):
            try:
                if start <= i <= end:
                    #print(f"Processing item {i}: {item['f']}")
                    
                    # Скачиваем изображение
                    image = download_image(item["f"])
                    if image is None:
                        print(f"Skipping item {i} due to download error.")
                        continue

                    # Проверяем, что изображение корректно
                    if not hasattr(image, "size") or not isinstance(image.size, tuple):
                        print(f"Invalid image size for item {i}. Skipping.")
                        continue

                    # Даунсемплим изображение
                    image = downscale_image_by(image,768,64)

                    folder = "gb768"

                    # Сохраняем изображение
                    output_path = f"{folder}/gb_{start}_{end}_{i}.jpg"
                        
                    image.save(output_path, quality=96)
                    #print(f"Saved image {i} to {output_path}")


                    # Удаляем переводы строк и заменяем _ на пробелы
                    processed_text = item['t'].replace("\n", " ").replace("_", " ").replace("'", "")

                    # Сохраняем в файл
                    output_file = f"{folder}/gb_{start}_{end}_{i}.txt"
                    with open(output_file, 'w', encoding='utf-8') as file:
                        file.write(processed_text)

                    #print(f"Текст сохранён в файл: {output_file}")

                elif i > end:
                    break

            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue

# Пример использования
iterate_range('hqdataset.txt', 0, 200000)
print('done')