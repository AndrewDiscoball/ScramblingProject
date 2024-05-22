import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image


def normalize_image(image):
    # Нормализация значений компонент изображения в интервал [0, 1]
    return np.array(image) / 255.0

def pad_image_to_block_size(image, block_size=8):
    height, width, channels = image.shape
    padded_height = np.ceil(height / block_size) * block_size
    padded_width = np.ceil(width / block_size) * block_size
    padded_image = np.zeros((int(padded_height), int(padded_width), channels))
    padded_image[:height, :width, :] = image
    return padded_image

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

def descramble_image(scrambled_image, seed, p, n):
    np.random.seed(seed)
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

def psnr(original_image, scrambled_image):
    mse = np.mean((original_image - scrambled_image) ** 2)
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Тестирование кода
image_path = input('Введите путь к файлу изображения: ')
original_image = np.array(Image.open(image_path))
original_image_padded = pad_image_to_block_size(original_image)
normalized_original_image = normalize_image(original_image_padded)

seed = 300  # Зерно генератора случайных чисел
p = 0.9  # Вероятность p
n = 2  # Число n
scrambled_image = scramble_image(normalized_original_image, seed, p, n)
scrambled_image = (scrambled_image * 255).astype(np.uint8)
scrambled_image = Image.fromarray(scrambled_image)
scrambled_image.save("scrambled_image.png")
normalized_scrambled_image = normalize_image(scrambled_image) 
 # Нормализация скремблированного изображения перед дескремблированием
descrambled_image = descramble_image(normalized_scrambled_image, seed, p, n)
descrambled_image = (descrambled_image * 255).astype(np.uint8)
descrambled_image = Image.fromarray(descrambled_image)
descrambled_image.save("descrambled_image.png")

psnr_value = psnr(normalized_original_image, normalized_scrambled_image)
print("PSNR:", psnr_value)
