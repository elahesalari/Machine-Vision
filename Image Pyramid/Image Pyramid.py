import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_pyramid(image):
    row, col = image.shape
    half_col = int(col * (1 / 2))
    temp = np.ones((image.shape[0], image.shape[1] + half_col))
    layer = image.copy()
    gaussian_pyrm = [layer]
    temp[0:row, 0:col] = layer
    for i in range(4):
        i += 1
        layer = cv2.pyrDown(layer)
        gaussian_pyrm.append(layer)
        plot_image(temp, layer, i)
    plt.imshow(temp, cmap='gray')
    plt.show()

    laplacian_pyramid(gaussian_pyrm, row, col, half_col)



def laplacian_pyramid(gaussian_pyrm, row, col, half_col):
    lp = []
    temp = np.zeros((row, col + half_col))
    plot_image(temp, gaussian_pyrm[4], 4)
    for i in range(4, 0, -1):
        gaussian_extended = cv2.pyrUp(gaussian_pyrm[i])
        gaussian_extended = cv2.resize(gaussian_extended, gaussian_pyrm[i-1].shape,interpolation = cv2.INTER_NEAREST)
        laplacian = cv2.subtract(gaussian_pyrm[i - 1], gaussian_extended)
        lp.append(laplacian)
        i -= 1
        plot_image(temp, laplacian, i)

    plt.imshow(temp, cmap='gray')
    plt.show()



def plot_image(temp, layer, i):
    row, col = image.shape
    half_col = int(col * (1 / 2))
    half_row = int(row * (1 / 2))
    quarter_row = int(row * (1 / 4))
    quarter_col = int(col * (1 / 4))
    eighth_row = int(row * (1 / 8))
    eight_col = int(col * (1 / 8))
    sixteenth_row = int(row * (1 / 16))
    sixteenth_col = int(col * (1 / 16))

    if i == 0:
        temp[0:row, 0:col] = layer
    elif i == 1:
        temp[0:half_row, col:] = layer
    elif i == 2:
        temp[half_row: half_row + quarter_row + 1, col: col + quarter_col + 1] = layer
    elif i == 3:
        temp[half_row:half_row + eighth_row + 1, col + quarter_col: col + quarter_col + eight_col + 1] = layer
    elif i == 4:
        temp[half_row + eighth_row: half_row + eighth_row + sixteenth_row + 1,
        col + quarter_col: col + quarter_col + sixteenth_col + 1] = layer


if __name__ == '__main__':
    image = cv2.imread('Lena.png')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gaussian_pyramid(image)
