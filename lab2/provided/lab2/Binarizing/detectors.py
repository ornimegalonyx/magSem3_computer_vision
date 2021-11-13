import numpy as np
import cv2


# Детектор по критерию идеального наблюдателя:
def optimal(img, s, mu):
    E = s ** 2  # энергия сигнала
    q = (np.double(img) - mu) * s  # корреляция
    h = E / 2  # пороговая величина
    imgo = np.uint8(q > h)  # решающее правило
    return imgo


# Детектор по критерию Неймана-Пирсона:
def Neyman_Pearson(img, mu, sigma):
    h = mu - 2 * sigma  # пороговая величина
    imgo = np.uint8(img > h)  # решающее правило
    return imgo


# Детектор по методу Оцу:
def Otsu(img):
    th, imgo = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # реализован в OpenCV
    return np.bool_(imgo)


# Адаптивный детектор по одному из алгоритмов:
def adaptive(img):

    # Параметры:
    R = 15               # размер апертуры
    rh = np.uint8(np.floor(R/2))

    # Сдвиговое окно:
    def rolling_window(a, shape):
        s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
        strides = a.strides + a.strides
        return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)

    windows = rolling_window(img, (R, R))

    # расчет порога для каждого положения окна:
    h = np.zeros(img.shape)
    for y in range(windows.shape[0]):
        for x in range(windows.shape[1]):
            aper = windows[y][x]
            mu = np.mean(aper)
            sigma = np.std(aper)
            k = 0.03
            # расчет порога:
            h[y+rh][x+rh] = mu + k * sigma

    # решающее правило:
    imgo = img > h

    return imgo
