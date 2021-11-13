import cv2
import numpy as np
import matplotlib.pyplot as plt
import detectors as detectors               # функции для бинаризации изображения
# функции для измерения ошибок бинаризации
import metrics as metrics
import show as show                         # функции для отрисовки
import os


# 0. Параметры:
TEST_IMG_FILE = os.path.join(os.path.dirname(__file__),
                             "../data/tst_img.png")     # идеальное изображение объекта
mu = 120                       # мат. ожидание аддитивного белого гауссовского шума
sigmas = np.arange(4, 30, 1)       # диапазон СКО шума для исследования


# 1. Константы:
# идеальное изображение без шума
ideal_img = np.uint8(cv2.imread(TEST_IMG_FILE, cv2.IMREAD_GRAYSCALE))
# размеры изображения
img_size = ideal_img.shape
# уровень сигнальных пикселей на идеальном изображении
s = ideal_img[0, 0]
# исследуемые отношения сигнал/шум в дБ
SNRs = 10 * np.log10((s ** 2) / (np.array(sigmas) ** 2))
# ошибки 1-го и 2-го рода для оптимального детектора
Ierr_img0 = IIerr_img0 = np.array([])
# ошибки 1-го и 2-го рода для детектора Неймана-Пирсона
Ierr_img1 = IIerr_img1 = np.array([])
# ошибки 1-го и 2-го рода для детектора Оцу
Ierr_img2 = IIerr_img2 = np.array([])
# ошибки 1-го и 2-го рода для адаптивного метода
Ierr_img3 = IIerr_img3 = np.array([])

# 2. Подготовка фигуры для вывода результатов бинаризации:
fig = plt.figure()
img_ax = show.axes_init([0.04, 0.54, 0.30, 0.40], "Original")
img_hist_ax = show.axes_init(
    [0.40, 0.54, 0.57, 0.40], "Original histogram", "on")
imgo0_ax = show.axes_init([0.04, 0.04, 0.20, 0.40], "Optimal detector")
imgo1_ax = show.axes_init([0.28, 0.04, 0.20, 0.40], "Neyman-Pearson")
imgo2_ax = show.axes_init([0.52, 0.04, 0.20, 0.40], "Otsu")
imgo3_ax = show.axes_init([0.76, 0.04, 0.20, 0.40], "Adaptive")
plt.show(block=False)

# 3. Цикл по всем возможным значениям сигнал/шум:
for i, sigma in enumerate(sigmas):

    # 3.1. Генерация тестового изображения:
    img = np.uint8(ideal_img + np.round(np.random.normal(mu, sigma, img_size)))

    # 3.2. Бинаризация изображения разными методами:
    # с помощью критерия идеального наблюдателя
    imgo0 = detectors.optimal(img, s, mu)
    # с помощью критерия Неймана-Пирсона
    imgo1 = detectors.Neyman_Pearson(img, mu+s, sigma)
    imgo2 = detectors.Otsu(img)                                 # методом Оцу
    # адаптивным методом
    imgo3 = detectors.adaptive(img)

    # 3.3. Вычисление ошибок 1-го и 2-го рода:
    Ierr_img0 = np.append(Ierr_img0, metrics.I_error(ideal_img, imgo0))
    IIerr_img0 = np.append(IIerr_img0, metrics.II_error(ideal_img, imgo0))
    Ierr_img1 = np.append(Ierr_img1, metrics.I_error(ideal_img, imgo1))
    IIerr_img1 = np.append(IIerr_img1, metrics.II_error(ideal_img, imgo1))
    Ierr_img2 = np.append(Ierr_img2, metrics.I_error(ideal_img, imgo2))
    IIerr_img2 = np.append(IIerr_img2, metrics.II_error(ideal_img, imgo2))
    Ierr_img3 = np.append(Ierr_img3, metrics.I_error(ideal_img, imgo3))
    IIerr_img3 = np.append(IIerr_img3, metrics.II_error(ideal_img, imgo3))

    # 3.4. Вывод изображений:
    show.img_update(img_ax, img, "Original")
    show.hist_update(img_hist_ax, img,
                     "Original histogram (SNR = %0.1f dB)" % SNRs[i])
    show.img_update(imgo0_ax, imgo0, "Optimal detector")
    show.img_update(imgo1_ax, imgo1, "Neyman-Pearson")
    show.img_update(imgo2_ax, imgo2, "Otsu")
    show.img_update(imgo3_ax, imgo3, "Adaptive")
    plt.pause(0.01)


# 4. Вывод графика зависимостей ошибок от отношения сигнал/шум:
# 4.1. Подготовка фигуры:
fig = plt.figure()
Ierr = show.axes_init([0.04, 0.75, 0.92, 0.20],
                      "Ошибка 1-го рода (захват пикселей фона)", ticks="on")
IIerr = show.axes_init([0.04, 0.40, 0.92, 0.20],
                       "Ошибка 2-го рода (пропуск пикселей сигнала)", ticks="on")
Allerr = show.axes_init([0.04, 0.05, 0.92, 0.20],
                        "Суммарная ошибка", ticks="on")

# 4.2. Вывод графика ошибок 1-го рода:
Ierr.plot(SNRs, Ierr_img0, label="optimal")
Ierr.legend(loc='upper right')
Ierr.plot(SNRs, Ierr_img1, label="Neyman-Pearson")
Ierr.legend(loc='upper right')
Ierr.plot(SNRs, Ierr_img2, label="Otsu")
Ierr.legend(loc='upper right')
Ierr.plot(SNRs, Ierr_img3, label="adaptive")
Ierr.legend(loc='upper right')
Ierr.set_ylim([0, 100])
Ierr.set_ylabel("Perr, %")
Ierr.set_xlim([0, SNRs.max()])
Ierr.set_xlabel("SNR, dB")
Ierr.grid()

# 4.3. Вывод графика ошибок 2-го рода:
IIerr.plot(SNRs, IIerr_img0, label="optimal")
IIerr.legend(loc='upper right')
IIerr.plot(SNRs, IIerr_img1, label="Neyman-Pearson")
IIerr.legend(loc='upper right')
IIerr.plot(SNRs, IIerr_img2, label="Otsu")
IIerr.legend(loc='upper right')
IIerr.plot(SNRs, IIerr_img3, label="adaptive")
IIerr.legend(loc='upper right')
IIerr.set_ylim([0, 100])
IIerr.set_ylabel("Perr, %")
IIerr.set_xlim([0, SNRs.max()])
IIerr.set_xlabel("SNR, dB")
IIerr.grid()

# 4.3. Вывод графика суммарных ошибок:
Allerr.plot(SNRs, Ierr_img0+IIerr_img0, label="optimal")
Allerr.legend(loc='upper right')
Allerr.plot(SNRs, Ierr_img1+IIerr_img1, label="Neyman-Pearson")
Allerr.legend(loc='upper right')
Allerr.plot(SNRs, Ierr_img2+IIerr_img2, label="Otsu")
Allerr.legend(loc='upper right')
Allerr.plot(SNRs, Ierr_img3+IIerr_img3, label="adaptive")
Allerr.legend(loc='upper right')
Allerr.set_ylim([0, 100])
Allerr.set_ylabel("Perr, %")
Allerr.set_xlim([0, SNRs.max()])
Allerr.set_xlabel("SNR, dB")
Allerr.grid()

# 4.4. Отображение графика:
plt.show()
