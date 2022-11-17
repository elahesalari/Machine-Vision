import numpy as np
import cv2
import imutils
import os
import matplotlib.pyplot as plt


def process_image():
    path = 'data'
    filenames = ['1.jpg', '2.jpg', '3.jpg']
    for filename in filenames:
        image = cv2.imread(os.path.join(path, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
        gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        blurred = cv2.blur(gradient, (27, 27))
        (_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # plt.imshow(closed, cmap='gray')
        # plt.show()

        closed = cv2.dilate(closed, None, iterations=8)  # Bold
        closed = cv2.erode(closed, None, iterations=4)  # Thin

        # plt.imshow(closed, cmap='gray')
        # plt.show()

        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        rect = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
        cv2.imwrite(f'results/{filename}.jpg',image)


if __name__ == '__main__':
    process_image()
