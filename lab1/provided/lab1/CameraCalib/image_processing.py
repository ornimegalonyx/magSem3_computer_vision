import numpy as np
import scipy.signal as s

# 1. Выделение углов клеток на изображении:
def corners_highlighting(imgi, hsize):

    # 1.1. Формируем ядро дифференциального оператора "[-1 1; 1 -1]" размером hsize:
    hsize_half = int(hsize / 2)
    h = np.array([[1] * hsize_half] * hsize_half)
    h = np.hstack([h, -1 * h])
    h = np.vstack([h, -1 * h])  # h = |hsize x hsize|

    # 1.2. Свертка с ядром:
    imgo = np.array(s.convolve2d(imgi, h, mode='same', boundary='wrap'))
    imgo = 255 * np.abs(imgo) / np.max(imgo)  # abs - крестики бывают положительные и отрицательные

    return imgo


# 2. Преобразуем выделенные точки в объекты:
def corners_to_objs(imgi, objnum, hsize):
    # 1.2.1. Создаем координатную сетку:
    y = range(0, imgi.shape[0])
    x = range(0, imgi.shape[1])
    x_grid, y_grid = np.meshgrid(x, y)

    # 1.2.2. Объединяем рядом стоящие пиксели начиная с максимального:
    objs = []
    for i in range(objnum):

        # 1.2.2.1. выбираем координаты максимального пикселя:
        p = np.unravel_index(imgi.argmax(), imgi.shape)  # p = (y, x)

        # 1.2.2.2. Добавляем все соседние пиксели к максимальному:
        top = np.maximum(p[0] - hsize, 0)
        btm = np.minimum(p[0] + hsize, imgi.shape[0])
        lft = np.maximum(p[1] - hsize, 0)
        rth = np.minimum(p[1] + hsize, imgi.shape[1])
        img_local = imgi[top:btm, lft:rth]
        y_local = y_grid[top:btm, lft:rth]
        x_local = x_grid[top:btm, lft:rth]
        if np.sum(img_local) != 0:
            point = [np.sum(img_local * y_local), np.sum(img_local * x_local)] / np.sum(img_local)
            objs.append(point)

        # 1.2.2.3. Удаляем учтенные пиксели с изображения:
        top = np.maximum(p[0] - hsize, 0)
        btm = np.minimum(p[0] + hsize, imgi.shape[0])
        lft = np.maximum(p[1] - hsize, 0)
        rth = np.minimum(p[1] + hsize, imgi.shape[1])
        imgi[top:btm, lft:rth] = 0

    return np.array(objs)
