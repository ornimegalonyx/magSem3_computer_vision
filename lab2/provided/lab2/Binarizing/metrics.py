import numpy as np

# Вычисление ошибки 1-го рода (ложный захват фоновых пикселей)
def I_error(ideal_img, bin_img):
    Cik = np.sum(ideal_img != 0)                # общее количество сигнальных пикселей на идеальном изображении
    TotalPix = np.prod(ideal_img.shape)         # общее количество пикселей на изображении
    Cki = np.sum(bin_img != 0)                  # количество пикселей, определенных как сигнальные
    Ckk = np.sum((ideal_img != 0) * bin_img)    # количество правильно определенных сигнальных пикселей

    Ierr = ((Cki - Ckk) / (TotalPix - Cik)) * 100

    return Ierr


# Вычисление ошибки 2-го рода (пропуск сигнальных пикселей):
def II_error(ideal_img, bin_img):

    Cik = np.sum(ideal_img != 0)                # общее количество сигнальных пикселей на идеальном изображении
    Ckk = np.sum((ideal_img != 0) * bin_img)    # количество правильно определенных сигнальных пикселей
    IIerr = ((Cik - Ckk) / Cik) * 100

    return IIerr

