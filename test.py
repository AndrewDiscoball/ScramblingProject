import numpy as np
from PIL import Image
from scramble import normalize_image, pad_image_to_block_size, scramble_image, descramble_image, psnr

def test_scramble_descramble():
    original_image = np.array(Image.open(Protorave.png))
    original_image_padded = pad_image_to_block_size(original_image)
    normalized_original_image = normalize_image(original_image_padded)

    seed = 300  # Зерно генератора случайных чисел
    p = 0.9  # Вероятность p
    n = 2  # Число n
    scrambled_image = scramble_image(normalized_original_image, seed, p, n)
    scrambled_image = (scrambled_image * 255).astype(np.uint8)
    scrambled_image = Image.fromarray(scrambled_image)
    scrambled_image.show()  # Показать скремблированное изображение
    scrambled_image.save("scrambled_image.png")
    normalized_scrambled_image = normalize_image(scrambled_image) 
     # Нормализация скремблированного изображения перед дескремблированием
    descrambled_image = descramble_image(normalized_scrambled_image, seed, p, n)
    descrambled_image = (descrambled_image * 255).astype(np.uint8)
    descrambled_image = Image.fromarray(descrambled_image)
    descrambled_image.show()  # Показать дескремблированное изображение
    descrambled_image.save("descrambled_image.png")

    psnr_value = psnr(normalized_original_image, normalized_scrambled_image)
    print("PSNR:", psnr_value)

if __name__ == "__main__":
    test_scramble_descramble()
