import cv2


def hsv_renge_claculator(h, s, v):
    return 180 * h / 360, 255 * s / 100, 255 * v / 100



if __name__ == "__main__":
    im = cv2.imread("data/hands.png", 1)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    start_color = hsv_renge_claculator(h=0, s=0, v=5)
    end_color = hsv_renge_claculator(h=40, s=70, v=100)
    mask = cv2.inRange(im_hsv, start_color, end_color)
    cv2.imwrite("result/part_one/gray_hand.jpg", mask)
