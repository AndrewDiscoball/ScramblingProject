## @file
## @brief Этот файл содержит функции для тестирования процесса скремблирования и дескремблирования изображений.

import numpy as np
from PIL import Image
from scramble import normalize_image, pad_image_to_block_size, scramble_image, descramble_image, psnr

def test_scramble_descramble():
    """
    Функция для тестирования процесса скремблирования и дескремблирования изображения.

    @return Ничего не возвращает.
    """
    original_image = np.array(Image.open("Protorave.png"))  # Открываем исходное изображение
    original_image_padded = pad_image_to_block_size(original_image)  # Делаем размер изображения кратным размеру блока
    normalized_original_image = normalize_image(original_image_padded)  # Нормализуем изображение

    seed = 300  # Зерно генератора случайных чисел
    p = 0.9  # Вероятность p
    n = 2  # Число n
    scrambled_image = scramble_image(normalized_original_image, seed, p, n)  # Скремблируем изображение
    scrambled_image = (scrambled_image * 255).astype(np.uint8)  # Преобразуем изображение обратно в 8-битное
    scrambled_image = Image.fromarray(scrambled_image)  # Преобразуем массив в изображение
    scrambled_image.show()  # Показываем скремблированное изображение
    scrambled_image.save("scrambled_image.png")  # Сохраняем скремблированное изображение
    normalized_scrambled_image = normalize_image(scrambled_image)  # Нормализуем скремблированное изображение перед дескремблированием
    descrambled_image = descramble_image(normalized_scrambled_image, seed, p, n)  # Дескремблируем изображение
    descrambled_image = (descrambled_image * 255).astype(np.uint8)  # Преобразуем изображение обратно в 8-битное
    descrambled_image = Image.fromarray(descrambled_image)  # Преобразуем массив в изображение
    descrambled_image.show()  # Показываем дескремблированное изображение
    descrambled_image.save("descrambled_image.png")  # Сохраняем дескремблированное изображение

    psnr_value = psnr(normalized_original_image, normalized_scrambled_image)  # Вычисляем PSNR между исходным и дескремблированным изображениями
    print("PSNR:", psnr_value)  # Выводим значение PSNR

if __name__ == "__main__":
    test_scramble_descramble()  # Запускаем функцию тестирования
