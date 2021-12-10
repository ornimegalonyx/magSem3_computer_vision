import cv2
import numpy as np


class Berry:

    def __init__(self, filename, group=None):
        self.load(filename)
        self.group = group

    def load(self, filename):
        self.filename = filename
        self.img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        self.feauters = Features(self.img)


class Features:

    def __init__(self, img):
        S = 0
        color = [0, 0, 0]
        for y in range(0, img.shape[0]):
            for x in range(0, img.shape[1]):
                if np.sum(img[y, x] - [255, 255, 255]) != 0:
                    S = S + 1
                    color = color + img[y, x]

        color = np.uint8(np.round(color / S))
        gray = np.uint8(np.round(0.2989 * color[0] + 0.5870 * color[1] + 0.1140 * color[2]))

        self.size = S / (img.shape[0] * img.shape[1])
        self.color_RGB = color
        self.color_gray = gray / 255.0
