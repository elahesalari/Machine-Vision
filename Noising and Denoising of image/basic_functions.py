from PIL import Image as im
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import random
import cv2
import math

def convert_image_to_array(img):
    return np.array(img)


def read_image(path, gray=True):
    img = im.open(path)
    if gray:
        img = img.convert('L')
    return convert_image_to_array(img)

def add_gaussian_noise(X_img):
    row, col = X_img.shape
    mu, sigma = 0, 0.1  # Centre of the distribution, Standard deviation of the distribution
    # creating a noise with the same dimension as the dataset (2,2)
    noise = 1000 * np.random.normal(mu, sigma, [row, col])
    noisy = X_img + noise

    return noisy

def add_salt_peper_noise(img):
    row, col = img.shape
    number_of_pixels = random.randint(3000, 10000)
    for _ in range(number_of_pixels):
        x_coord = random.randint(0, row - 1)
        y_coord = random.randint(0, col - 1)
        img[x_coord][y_coord] = 255

    number_of_pixels = random.randint(3000, 10000)
    for _ in range(number_of_pixels):
        x_coord = random.randint(0, row - 1)
        y_coord = random.randint(0, col - 1)
        img[x_coord][y_coord] = 0

    return img


def calculate_psnr(img1, img2):   # Peak Signal-to-Noise Ratio
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score = ssim(img1, img2)
    return score
