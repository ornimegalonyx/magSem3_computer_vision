import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as s
import math
import os

os.system('cls')
# 0. Параметры:
IMG_FILE = os.getcwd().replace('\\', '/') + "/lab1/input/var16/15cm_12MP.jpg"
PIXEL_SIZE_MM = 2 * 0.8 * 10 ** -3
DISTANSE_MM = 150
FOCUS_MM = 4.94
LowPassCore = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 2, 3, 2, 1],
        [1, 2, 2, 2, 1],
        [1, 1, 1, 1, 1]
    ]
)/35
HighPassCore = np.array(
    [
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ]
)

# 1. Считываем файл:
imgi = np.uint8(cv2.imread(IMG_FILE, cv2.IMREAD_GRAYSCALE))

# 2. Отображаем файл:
fig = plt.figure(-1)
plt.imshow(imgi, cmap='gray', vmin=np.min(0), vmax=np.max(imgi))
plt.draw()
plt.show(block=False)
fig.canvas.draw()
fig.canvas.flush_events()

# lowPassFiltered = np.array(s.convolve2d(imgi, np.flip(
#     LowPassCore), mode='same', boundary='wrap'))
# filtered = np.array(s.convolve2d(lowPassFiltered, np.flip(
#     HighPassCore), mode='same', boundary='wrap'))

# 3. Формируем ядра свертки для поиска углов:
hsize = 7
h = np.array([[1] * hsize] * hsize)
hsize_half = int(np.floor(hsize/2))

tmp = h.copy()
tmp[0:hsize_half+1, hsize_half:] = -1
# tmp[tmp < 0] = -int(abs((255 - np.sum(tmp[tmp > 0]))/np.sum(tmp[tmp < 0])))
# hs.append(tmp/np.sum(np.sum(tmp)))
hs = [tmp]

tmp = h.copy()
tmp[np.tril_indices(hsize)] = -3
tmp[0:, 0:hsize_half] = 1
# tmp[tmp < 0] = -int(abs((255 - np.sum(tmp[tmp > 0]))/np.sum(tmp[tmp < 0])))
# hs.append(tmp/np.sum(np.sum(tmp)))
hs.append(tmp)

tmp = h.copy()
tmp[np.tril_indices(hsize)] = -3
tmp[hsize_half+1:, 0:] = 1
# tmp[tmp < 0] = -int(abs((255 - np.sum(tmp[tmp > 0]))/np.sum(tmp[tmp < 0])))
# hs.append(tmp/np.sum(np.sum(tmp)))
hs.append(tmp)

# 4. Ищем каждый угол:
p = []
for i in range(len(hs)):
    # print(hs[i], '\n')

    # 4.1. Свертка с ядром для угла:
    imgo = np.array(s.convolve2d(imgi, np.flip(
        hs[i]), mode='same', boundary='wrap'))
    # plt.figure(i)
    # plt.imshow(imgo, cmap='gray')
    # plt.draw()
    # plt.show(block=False)
    # fig.canvas.draw()
    # fig.canvas.flush_events()

    # 4.2. Находим пиксель максимальной яркости:
    p.append(np.unravel_index(imgo.argmax(), imgo.shape))

    # 4.3. Отображаем найденный пиксель:
    plt.text(p[-1][1], p[-1][0], 'o', color="green")
    fig.canvas.draw()
    fig.canvas.flush_events()

p_bl = p[0]
p_t = p[1]
p_br = p[2]

# 5. Считаем длину рёбер (расстояние между точками):
length_bl_t = np.sqrt((p_bl[0] - p_t[0])**2 + (p_bl[1] - p_t[1])**2)
length_t_br = np.sqrt((p_t[0] - p_br[0])**2 + (p_t[1] - p_br[1])**2)
length_bl_br = np.sqrt((p_bl[0] - p_br[0])**2 + (p_bl[1] - p_br[1])**2)

# 6. Считаем периметр:
perimeter = length_bl_t + length_t_br + length_bl_br
perimeter_mm = (perimeter * PIXEL_SIZE_MM) * DISTANSE_MM / FOCUS_MM

# 7. Длина ребер в мм:
length_bl_t_mm = (length_bl_t * PIXEL_SIZE_MM) * DISTANSE_MM / FOCUS_MM
length_t_br_mm = (length_t_br * PIXEL_SIZE_MM) * DISTANSE_MM / FOCUS_MM
length_bl_br_mm = (length_bl_br * PIXEL_SIZE_MM) * DISTANSE_MM / FOCUS_MM

# 8. Угловые размеры фигуры:
AngleY = math.atan((p_t[0] - p_br[0]) *
                   PIXEL_SIZE_MM / FOCUS_MM) * 180 / math.pi
AngleX = math.atan((p_br[1] - p_t[1]) *
                   PIXEL_SIZE_MM / FOCUS_MM) * 180 / math.pi

# 9. Площадь фигуры

area = length_bl_t_mm * length_bl_br_mm / 2

# 10. Выводим результаты в консоль:
print("Perimeter length = %d pixs (%0.2f mm)" % (perimeter, perimeter_mm))
print("L0 = %d pixs (%0.2f mm)" % (length_bl_t, length_bl_t_mm))
print("L1 = %d pixs (%0.2f mm)" % (length_t_br, length_t_br_mm))
print("L2 = %d pixs (%0.2f mm)" % (length_bl_br, length_bl_br_mm))
print("Area = %0.2f mm^2" % area)
print("AngleY = %0.2f °" % AngleY)
print("AngleX = %0.2f °" % AngleX)
input("Press ENTER...")
