## @file
## @brief Этот файл содержит функции для скремблирования и дескремблирования изображений.

import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image

## Нормализует значения компонент изображения в интервал [0, 1]
# @param image Изображение для последующей нормализации.
# @return Нормализованное изображение.
def normalize_image(image):
    return np.array(image) / 255.0

## Дополняет изображение до размера блока
# @param image Изображение для дополнения.
# @param block_size Размер блока для дополнения.
# @return Дополненное изображение.
def pad_image_to_block_size(image, block_size=8):
    height, width, channels = image.shape
    padded_height = np.ceil(height / block_size) * block_size
    padded_width = np.ceil(width / block_size) * block_size
    padded_image = np.zeros((int(padded_height), int(padded_width), channels))
    padded_image[:height, :width, :] = image
    return padded_image

## Скремблирует изображение
# @param image Изображение для скремблирования.
# @param seed Зерно для генератора случайных чисел.
# @param p Вероятность появления 1 в матрице Bk.
# @param n Порог для модификации матрицы DCT.
# @return Скремблированное изображение.
def scramble_image(image, seed, p, n):
    np.random.seed(seed)
    height, width, channels = image.shape
    scrambled_image = np.zeros_like(image)

    # Разбиваем изображение на блоки 8x8
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            for c in range(channels):
                block = image[i:i + 8, j:j + 8, c]
                # Применяем дискретное косинусное преобразование (DCT)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

                # Создаем матрицу Bk с элементами -1 и 1
                Bk = np.random.choice([1, -1], size=(8, 8), p=[p, 1 - p])

                # Модифицируем матрицу DCT
                for x in range(8):
                    for y in range(8):
                        if x >= n or y >= n:
                            dct_block[x, y] *= Bk[x, y]

                # Применяем обратное дискретное косинусное преобразование (IDCT)
                idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

                # Ограничиваем значения в пределах [0, 1]
                idct_block[idct_block < 0] = 0
                idct_block[idct_block > 1] = 1

                scrambled_image[i:i + 8, j:j + 8, c] = idct_block

    return scrambled_image

## Дескремблирует изображение
# @param scrambled_image Скремблированное ранее изображение.
# @param seed Зерно для генератора случайных чисел.
# @param p Вероятность появления 1 в матрице Bk.
# @param n Порог для модификации матрицы DCT.
# @return Дескремблированное изображение изображение.
def descramble_image(scrambled_image, seed, p, n):
    np.random.seed(seed+9)
    height, width, channels = scrambled_image.shape
    descrambled_image = np.zeros_like(scrambled_image)

    # Разбиваем изображение на блоки 8x8
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            for c in range(channels):
                block = scrambled_image[i:i + 8, j:j + 8, c]
                # Применяем дискретное косинусное преобразование (DCT)
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

                # Создаем матрицу Bk с элементами -1 и 1
                Bk = np.random.choice([1, -1], size=(8, 8), p=[p, 1 - p])

                # Модифицируем матрицу DCT обратно
                for x in range(8):
                    for y in range(8):
                        if x >= n or y >= n:
                            dct_block[x, y] /= Bk[x, y]

                # Применяем обратное дискретное косинусное преобразование (IDCT)
                idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

                # Ограничиваем значения в пределах [0, 1]
                idct_block[idct_block < 0] = 0
                idct_block[idct_block > 1] = 1

                descrambled_image[i:i + 8, j:j + 8, c] = idct_block

    return descrambled_image

## Ищет отношение сигнал/шум у исходного и скремблированного изображений
def psnr(original_image, scrambled_image):
    mse = np.mean((original_image - scrambled_image) ** 2)
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))
