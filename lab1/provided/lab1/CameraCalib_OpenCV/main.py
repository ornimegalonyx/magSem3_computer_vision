import cv2
import numpy as np
import glob
import math
import os

# 0. Константы:
# 0.1. Изображение:
IMGs_PATH = os.path.join(os.path.dirname(__file__),
                         "../../../input/chessboard/48MP/*.jpg")  # путь к изображениям шахматной доски
# количество уголков между клетками по (вертикали, горизонтали)
CHECKERBOARD = (6, 9)

# 0.2. Параметры камеры:
# размер пикселя по [вертикали, горизонтали]
PIXEL_SIZES_MM = [0.8 * 10 ** -3, 0.8 * 10 ** -3]

# 0.3. Настройки для программного кода:
# настройки поиска координат с субпиксельной точностью
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# 1. Инициализация переменных:
# вектор для хранения векторов 3D точек для каждого изображения шахматной доски
objpoints = []
# вектор для хранения векторов 2D точек для каждого изображения шахматной доски
imgpoints = []
# условные положения точек в 3D
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)    # [[x, y, z], ...]


# 2. Обработка изображений:
images = glob.glob(IMGs_PATH)       # список изображений
for fname in images:

    # 2.1. Считываем изображение в серых тонах:
    img = cv2.imread(fname)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2.2. Проводим первичный поиск уголков шахматной доски на изображении:
    # ret - флаг успеха поиска и упорядочивания всех уголков
    # corners - грубое значения кординат каждого уголка [y, x]
    ret, corners = cv2.findChessboardCorners(img_gray, CHECKERBOARD,    cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    # 2.3. Если поиск прошел неуспешно - переходим к следующему изображению:
    if ret is False:
        continue

    # 2.5. Уточняем координаты уголков:
    corners2 = cv2.cornerSubPix(
        img_gray, corners, (11, 11), (-1, -1), criteria)

    # 2.6. Добавляем найденные координаты в общий вектор:
    imgpoints.append(corners2)
    objpoints.append(objp)

    # 2.7. Отображаем изображение и найденные углы:
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    cv2.imshow('img', cv2.resize(img, (960, 540)))

    # 2.8. Ждем подтверждения от пользователя (нажатие любой клавиши):
    cv2.waitKey(0)


# 3. Закрываем все окна с изображениями:
cv2.destroyAllWindows()


# 4. Проводим калибровку камеры:
ret, K, dist_coef, R, T = cv2.calibrateCamera(
    objpoints, imgpoints, img_gray.shape[::-1], None, None)
if ret is False:
    print("Camera calibration fail")
    exit(-1)

# 5. Переводим найденные значения в миллиметры:
focus_mm = np.array([K[1][1], K[0][0]]) * PIXEL_SIZES_MM
center_mm = np.array([K[1][2], K[0][2]]) * PIXEL_SIZES_MM

# 6. Рассчитываем уголы обзора по вертикали и горизонтали:
FOV = np.array([0, 0])
FOV[0] = 2 * math.atan(center_mm[0] / focus_mm[0]) * 180 / math.pi
FOV[1] = 2 * math.atan(center_mm[1] / focus_mm[1]) * 180 / math.pi

# 7. Выводим результат:
print("Focus (mm): %f (y), %f (x)" % (focus_mm[0], focus_mm[1]))
print("Field of view (°): %d (y), %d (x)" % (FOV[0], FOV[1]))


# 8. Сохраняем результат в файл:
camera = {
    "PIXEL_SIZES_MM": PIXEL_SIZES_MM,
    "FOCUSES_MM": focus_mm,
    "FOVs": FOV,
    "DIST_COEF": dist_coef
}
file = open(os.path.join(os.path.dirname(__file__),
                         "../../../output/camera_OpenCV.txt"), "w")
file.write("%s = %s\n" % ("camera", camera))
file.close()


# 6. Ожидаем действия пользователя для выхода (нажатие любой клавиши):
cv2.waitKey(0)
