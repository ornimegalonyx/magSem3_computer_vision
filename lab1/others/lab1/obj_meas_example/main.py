import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as s
import math

# 0. Параметры:
IMG_FILE        = "../data/Figure1_4.jpg"
PIXEL_SIZE_MM   = 1.7*10**-3
DISTANSE_MM     = 160
FOCUS_MM        = 5.4

# 1. Считываем файл:
imgi = np.uint8(cv2.imread(IMG_FILE, cv2.IMREAD_GRAYSCALE))

# 2. Отображаем файл:
fig = plt.figure()
plt.imshow(imgi, cmap='gray', vmin=np.min(0), vmax=np.max(imgi))
plt.draw()
plt.show(block=False)
fig.canvas.draw()
fig.canvas.flush_events()

# 3. Формируем ядра свертки для поиска углов:
hsize = 21
h = np.array([[1] * hsize] * hsize)
hsize_half = int(np.floor(hsize/2))
h[hsize_half:, 0:hsize_half+1] = -1             # type of h: [[1, 1], [-1, 1]]
hs = np.array([h, np.rot90(h, k=1), np.rot90(h, k=2), np.rot90(h, k=3)])

# 4. Ищем каждый угол:
p = []
for i in range(len(hs)):

    # 4.1. Свертка с ядром для угла:
    imgo = np.array(s.convolve2d(imgi, np.flip(hs[i]), mode='same', boundary='wrap'))

    # 4.2. Находим пиксель максимальной яркости:
    p.append(np.unravel_index(imgo.argmax(), imgo.shape))

    # 4.3. Отображаем найденный пиксель:
    plt.text(p[-1][1], p[-1][0], 'o', color="green")
    fig.canvas.draw()
    fig.canvas.flush_events()

p_tr = p[0]   # top right   [y, x]
p_tl = p[1]   # top left    [y, x]
p_bl = p[2]   # btm left    [y, x]
p_br = p[3]   # btm right   [y, x]

# 5. Считаем длины линий периметра:
perim_trtl = np.sqrt((p_tr[0] - p_tl[0])**2 + (p_tr[1] - p_tl[1])**2)
perim_tlbl = np.sqrt((p_tl[0] - p_bl[0])**2 + (p_tl[1] - p_bl[1])**2)
perim_blbr = np.sqrt((p_bl[0] - p_br[0])**2 + (p_bl[1] - p_br[1])**2)
perim_brtr = np.sqrt((p_br[0] - p_tr[0])**2 + (p_br[1] - p_tr[1])**2)

# 6. Считаем длину периметра:
perim = np.sum([perim_trtl, perim_tlbl, perim_blbr, perim_brtr])
perim_mm = (perim * PIXEL_SIZE_MM) * DISTANSE_MM / FOCUS_MM

# 7. Длина ребер:
l0 = np.mean([perim_trtl, perim_blbr])
l1 = np.mean([perim_tlbl, perim_brtr])
l0_mm = (l0 * PIXEL_SIZE_MM) * DISTANSE_MM / FOCUS_MM
l1_mm = (l1 * PIXEL_SIZE_MM) * DISTANSE_MM / FOCUS_MM

# 8. Угловые размеры фигуры:
AngleY = math.atan((p_bl[0] - p_tr[0]) * PIXEL_SIZE_MM / FOCUS_MM) * 180 / math.pi
AngleX = math.atan((p_tr[1] - p_bl[1]) * PIXEL_SIZE_MM / FOCUS_MM) * 180 / math.pi

# 9. Выводим результаты в консоль:
print("Perimeter length = %d pixs (%0.2f mm)" % (perim, perim_mm))
print("L0 = %d pixs (%0.2f mm)" % (l0, l0_mm))
print("L1 = %d pixs (%0.2f mm)" % (l1, l1_mm))
print("AngleY = %0.2f °" % AngleY)
print("AngleX = %0.2f °" % AngleX)

# 10. Ждем сигнала от пользователя для выхода:
input("Press Enter to exit...")
