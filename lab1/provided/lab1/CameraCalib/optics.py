import math
import numpy as np
import scipy.ndimage as sp
import matplotlib.pyplot as plt

# 0. Расчет средней коориданты по столбцам точек:
def mean_coords(objs_coord, lim):

    # 0.1. Сортируем координаты точек по возрастанию
    coords = objs_coord[objs_coord.argsort()]

    # 0.2. Разделяем координаты на группы по расстоянию lim:
    mean_coords = []
    while len(coords) != 0:
        # 0.2.1. Вынимаем группу точек из общего массива:
        group = coords[coords < (coords[0] + lim)]
        coords = coords[len(group):]

        # 0.2.2. Считаем среднее по группе точек и добавляем в выходные значения:
        mean_coords.append(np.mean(group))

    return np.array(mean_coords)

# 1. Построение идеальной сетки точек по измеренным точкам:
def objs_ideal (objs, lims):

    # 1.1. Считаем средние координаты для строк и столбцов:
    y_base = mean_coords(objs[:, 0], lims[0])
    x_base = mean_coords(objs[:, 1], lims[1])

    # 1.2. Формируем массив идеальных объектов:
    xm, ym = np.meshgrid(x_base, y_base)
    objs_ideal = np.array([ym.ravel(), xm.ravel()]).T

    return objs_ideal


def objs_matching(objs_real, objs_ideal, lims):

    objs_r_out = []
    objs_i_out = []
    for n in range(len(objs_real)):
        if objs_real[n][0] != -1:
            for k in range(len(objs_ideal)):
                if objs_ideal[k][0] != -1:
                    if np.abs(objs_real[n][0] - objs_ideal[k][0]) < lims[0]:
                        if np.abs(objs_real[n][1] - objs_ideal[k][1]) < lims[1]:
                            objs_r_out.append(objs_real[n].copy())
                            objs_i_out.append(objs_ideal[k].copy())
                            objs_real[n] = np.array([-1, -1])
                            objs_ideal[k] = np.array([-1, -1])

    return np.array(objs_r_out), np.array(objs_i_out)

# 2. Нормировка координат объектов:
def norm(objs, center):

    objs_norm = np.copy(objs)
    objs_norm[:, 0] = (objs_norm[:, 0] - center[0])
    objs_norm[:, 1] = (objs_norm[:, 1] - center[1])

    return objs_norm


# 3. Получение коэффициентов радиальной дисторсии:
def dist_coeff(objs_real, objs_ideal, shape):

    # 3.1. Находим координаты центра и нормируем координаты:
    center = np.array(shape) / 2
    objs_real_norm = norm(objs_real, center)
    objs_ideal_norm = norm(objs_ideal, center)

    # 3.2. Считаем расстояние до каждой точки из центра:
    r_reals = np.sqrt(objs_real_norm[:, 0] ** 2 + objs_real_norm[:, 1] ** 2)
    r_ideals = np.sqrt(objs_ideal_norm[:, 0] ** 2 + objs_ideal_norm[:, 1] ** 2)

    # 3.3. Решаем матричное уравнение:
    r_ideals_matrix = np.matrix([r_ideals ** 2, r_ideals ** 4, r_ideals ** 6]).T
    results = np.matrix([(r_reals / r_ideals - 1).ravel()]).T
    coef = np.array(np.linalg.lstsq(r_ideals_matrix, results, rcond=None)[0])

    # 3.4. Отображаем скорректированные координаты с точками:
    objs_corr = dist_objs_correction(objs_real, shape, coef)
    for i in range(len(objs_corr)):
        plt.text(objs_corr[i][1], objs_corr[i][0], 'x', color="green")

    # 3.5. Формируем сетку, искаженныую дисторсией, для визуального контроля:
    y_v = np.linspace(0, shape[0], 50)
    x_v = np.linspace(0, shape[1], 50)
    x_m, y_m = np.meshgrid(x_v, y_v)
    tstpoints = np.array([y_m.ravel(), x_m.ravel()]).T
    tstpoints = dist_objs_correction(tstpoints, shape, coef, dir='reverse')

    return coef, tstpoints


# 4. Корректировка положения объектов:
def dist_objs_correction(objs, shape, coef, dir='forward'):

    # 4.0. Считаем положение центра:
    center = np.array(shape) / 2

    # 4.1. Нормируем координаты:
    objs = norm(objs, center)

    # 4.2. Считаем радиусы:
    radiuses = np.sqrt(objs[:, 0] ** 2 + objs[:, 1] ** 2)

    # 4.3. Считаем поправочные коэффициенты:
    mr = 1 + coef[0] * (radiuses ** 2) + coef[1] * (radiuses ** 4) + coef[2] * (radiuses ** 6)

    # 4.4. Корректируем координаты:
    if dir=='forward':
        objs[:, 0] = (objs[:, 0] / mr) + center[0]
        objs[:, 1] = (objs[:, 1] / mr) + center[1]
    else:
        objs[:, 0] = (objs[:, 0] * mr) + center[0]
        objs[:, 1] = (objs[:, 1] * mr) + center[1]

    return objs


# 5. Корректировка изображения:
def dist_img_correction(img, coef):

    # 5.1. Получаем размер изображения и его центр:
    shape = img.shape
    center = np.array(shape) / 2

    # 5.2. Формируем сетку координат:
    xf, yf = np.meshgrid(np.float32(np.arange(shape[1])), np.float32(np.arange(shape[0])))
    coord = np.array([yf.ravel(), xf.ravel()]).T
    coord = dist_objs_correction(coord, shape, coef, dir='reverse')

    # 5.3. Корректируем изображение:
    imgo = sp.map_coordinates(img, [coord[:, 0], coord[:, 1]])
    imgo.resize((3024, 4032))

    return imgo


# 6. Вычисление фокусного расстояния:
def focus_calc(objs, shape, PIXEL_SIZES, DISTANCE, SQUARE_SIZES, lims):

    # 6.1. Считаем средние координаты для строк и столбцов:
    y_base = mean_coords(objs[:, 0], lims[0])
    x_base = mean_coords(objs[:, 1], lims[1])

    # 6.2. Находим две ближайшие координаты к центру по x и по y:
    center = np.array(shape) / 2
    yb = y_base[np.abs(y_base - center[0]).argsort()[:2]]
    xb = x_base[np.abs(x_base - center[1]).argsort()[:2]]

    # 6.3. Вычисляем расстояния по x и по y (каждое расстояние - ребро клетки):
    dy = np.abs(yb[0] - yb[1])
    dx = np.abs(xb[0] - xb[1])

    # 6.4. Вычисляем фокусное расстояние:
    fy = ((dy * PIXEL_SIZES[0]) / SQUARE_SIZES[0]) * DISTANCE
    fx = ((dx * PIXEL_SIZES[1]) / SQUARE_SIZES[1]) * DISTANCE

    return fy, fx

# 7. Вычисление угла зрения:
def FOV_calc(shape, PIXEL_SIZES_MM, focus_mm):

    center = np.array(shape) / 2
    FOVy = 2 * math.atan(center[0] * PIXEL_SIZES_MM[0] / focus_mm[0]) * 180 / math.pi
    FOVx = 2 * math.atan(center[1] * PIXEL_SIZES_MM[1] / focus_mm[1]) * 180 / math.pi

    return FOVy, FOVx