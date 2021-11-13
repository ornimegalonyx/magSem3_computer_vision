from unittest import result
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import sinc
import scipy.signal as s
import math
import os
import sys
from scipy.spatial import distance


# np.set_printoptions(threshold=sys.maxsize)

os.system('cls')
# 0. Параметры:
IMG_FILE = os.getcwd().replace('\\', '/') + "/lab1/input/var16/26cm_12MP.jpg"
PIXEL_SIZE_MM = 2 * 0.8 * 10 ** -3
DISTANCE_MM = 260
FOCUS_MM = 4.94
# LowPassCore = np.array(
#     [
#         [1, 1, 1, 1, 1, 1, 1],
#         [1, 2, 2, 2, 2, 2, 1],
#         [1, 2, 3, 3, 3, 2, 1],
#         [1, 2, 3, 4, 3, 2, 1],
#         [1, 2, 3, 3, 3, 2, 1],
#         [1, 2, 2, 2, 2, 2, 1],
#         [1, 1, 1, 1, 1, 1, 1]
#     ]
# )
# LowPassCore = LowPassCore/np.sum(np.sum(LowPassCore))


def LP_filter(x, y):
    return np.sinc(x)*np.sinc(y)


# 1. Считываем файл:
imgi = np.uint8(cv2.imread(IMG_FILE, cv2.IMREAD_GRAYSCALE))

# 2. Отображаем файл:
fig0 = plt.figure('Оригинал')
plt.imshow(imgi, cmap='gray', vmin=np.min(0), vmax=np.max(imgi))
plt.draw()
plt.show(block=False)
fig0.canvas.draw()
fig0.canvas.flush_events()

# # 2.1 ФНЧ
# a = 1
# n = 51
# x = np.linspace(-a, a, n)
# y = np.linspace(-a, a, n)
# LowPassCore = LP_filter(x[:, None], y[None, :])
# print('Изображение фильтруется с помощью ФНЧ', n, 'x', n)
# # imgi = np.array(s.convolve2d(np.double(imgi), LowPassCore,
# #                 mode='same', boundary='wrap'))
# imgi = s.fftconvolve(np.double(imgi), LowPassCore, mode='same')
# imgi = 255 * (imgi/imgi.max())
# # imgi = cv2.GaussianBlur(imgi, (n, n), 0)
# fig_ = plt.figure('ФНЧ')
# plt.imshow(imgi, cmap='gray', vmin=np.min(0), vmax=np.max(imgi))
# plt.draw()
# plt.show(block=False)
# fig.canvas.draw()
# fig.canvas.flush_events()

# 3. Формируем ядра свертки для поиска точек:
hsize = 80
width = 23
border = (hsize-width)//2
h = np.array([[-1] * hsize] * hsize)
h[border:border+width, :] = 1
h[:, border:border+width] = 1

p = []
print('Производится двумерная свёртка с окном "звезда"', hsize, 'x', hsize)
# imgo = np.array(s.convolve2d(np.double(imgi), np.flip(h),
#                              mode='same', boundary='wrap'))
imgo = s.fftconvolve(np.double(imgi), h, mode='same')
imgo = 255 * (imgo / imgo.max())
imgo[imgo < 0] = 0
imgo[imgo > 255] = 255
fig1 = plt.figure('Свёртка со звездой')
plt.imshow(imgo, cmap='gray', vmin=np.min(0), vmax=np.max(imgo))
plt.draw()
plt.show(block=False)
fig1.canvas.draw()
fig1.canvas.flush_events()

hsize = 60
a = hsize//2
b = hsize//2
radius = 14
h = np.array([[-1] * hsize] * hsize)
for y in range(hsize):
    for x in range(hsize):
        if (x-a)**2 + (y-b)**2 < radius**2:
            h[x][y] = 1

print('Производится двумерная свёртка с окном "круг"', hsize, 'x', hsize)
# imgo = np.array(s.convolve2d(np.double(imgi), np.flip(h),
#                              mode='same', boundary='wrap'))
imgo = s.fftconvolve(np.double(imgo), h, mode='same')
imgo = 255 * (imgo / imgo.max())
imgo[imgo < 0] = 0
imgo[imgo > 255] = 255
fig2 = plt.figure('Свёртка с кругом')
plt.imshow(imgo, cmap='gray', vmin=np.min(0), vmax=np.max(imgo))
plt.draw()
plt.show(block=False)
fig2.canvas.draw()
fig2.canvas.flush_events()

hsize = 61
h = np.array([[-1] * hsize] * hsize)
h[(hsize-1)//2, (hsize-1)//2] = 255

print('Производится двумерная свёртка с окном "точка"', hsize, 'x', hsize)
# imgo = np.array(s.convolve2d(np.double(imgi), np.flip(h),
#                              mode='same', boundary='wrap'))
imgo = s.fftconvolve(np.double(imgo), h, mode='same')
imgo = 255 * (imgo / imgo.max())
imgo[imgo < 0] = 0
imgo[imgo > 255] = 255
fig2 = plt.figure('Свёртка с точкой')
plt.imshow(imgo, cmap='gray', vmin=np.min(0), vmax=np.max(imgo))
plt.draw()
plt.show(block=False)
fig2.canvas.draw()
fig2.canvas.flush_events()
p = np.argwhere(imgo > 150)
# print(p, len(p))
points = []

flag = 0
next_i = 0
for i in range(0, len(p)):
    index = []
    for j in range(0, len(p)):
        if (p[i][0]-p[j][0])**2 + (p[i][1]-p[j][1])**2 < radius**2:
            index.append([True, True])
        else:
            if flag == 0:
                next_i = j
                flag = 1
            index.append([False, False])
    i = j
    flag = 0
    tmp = np.reshape(p[index], (-1, 2))
    points.append(np.floor(np.mean(tmp, axis=0)))
points = np.unique(points, axis=0)
print('\nНайдено', len(points), 'звёзд:')
for i in range(0, len(points)):
    print('Точка', i, ':', points[i])

# 4 Поиск кратчайшего пути

len0to2 = np.sqrt((points[0][0] - points[2][0]) **
                  2 + (points[0][1] - points[2][1])**2)

len2to5 = np.sqrt((points[2][0] - points[5][0]) **
                  2 + (points[2][1] - points[5][1])**2)

len5to3 = np.sqrt((points[5][0] - points[3][0]) **
                  2 + (points[5][1] - points[3][1])**2)

len3to1 = np.sqrt((points[3][0] - points[1][0]) **
                  2 + (points[3][1] - points[1][1])**2)

len1to4 = np.sqrt((points[1][0] - points[4][0]) **
                  2 + (points[1][1] - points[4][1])**2)

len4to6 = np.sqrt((points[4][0] - points[6][0]) **
                  2 + (points[4][1] - points[6][1])**2)

resultLength = len0to2+len2to5+len5to3+len3to1+len1to4+len4to6
resultLength_mm = (resultLength * PIXEL_SIZE_MM) * DISTANCE_MM / FOCUS_MM

print('\nДлина пути в пикселях:', resultLength)
print('Длина пути в мм:', resultLength_mm)

input("\nPress ENTER...")
