import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from skimage.metrics import structural_similarity as ssim


def read_mages():
    path = 'data'
    angles = [30, 45]
    for filename in os.listdir(path):
        image = cv2.imread(os.path.join(path, filename))
        shifted_img = shift_image(image)
        mse_shift, psnr_shift = calculate_psnr(image, shifted_img)
        ssim_shift = calculate_ssim(image, shifted_img)
        mse_rotates, psnr_rotates, ssim_rotates = [], [], []
        for a in angles:
            rotated_image = rotate_image(image, a)
            mse_rotate, psnr_rotate = calculate_psnr(image, rotated_image)
            mse_rotates.append(mse_rotate)
            psnr_rotates.append(psnr_rotate)
            ssim_rotate = calculate_ssim(image, rotated_image)
            ssim_rotates.append(ssim_rotate)
        print('The filename: ', filename)
        print('------------------------')
        print('MSE Main-Shift: ', mse_shift)
        print('PSNR Main-Shift: ', psnr_shift)
        print('SSIM Main-Shift: ', ssim_shift)
        print('----------***-----------')
        print('MSE Main-Rotate-30: ', ssim_rotates[0])
        print('PSNR Main-Rotate-30: ', ssim_rotates[0])
        print('SSIM Main-Rotate-30: ', ssim_rotates[0])
        print('MSE Main-Rotate-45: ', ssim_rotates[1])
        print('PSNR Main-Rotate-45: ', ssim_rotates[1])
        print('SSIM Main-Rotate-45: ', ssim_rotates[1])
        print('----------///***///------------')


def shift_image(image):
    dx, dy = image.shape[0], image.shape[1]
    ext_x, ext_y = int(dx * (1 / 5)), int(dy * (1 / 6))
    # new_x, new_y = int(dx + ext_x), int(dy + ext_y)
    shifted = np.zeros((dx, dy, 3), dtype=image.dtype)
    shifted[ext_x:, ext_y:] = image[0:dx - ext_x, 0:dy - ext_y]
    plt.imshow(cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB))
    plt.show()
    return shifted


def rotate_image(image, angle):
    row, col = image.shape[0], image.shape[1]
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=0.8)
    rotated_image = cv2.warpAffine(image, rot_mat, (col, row))
    plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    plt.show()
    return rotated_image


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return mse, 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):  # Structural Similarity Index
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score = ssim(img1, img2)
    return score


if __name__ == '__main__':
    read_mages()
