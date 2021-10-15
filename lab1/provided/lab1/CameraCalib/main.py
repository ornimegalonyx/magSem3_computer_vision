import cv2
import numpy as np
import image_processing as improc
import optics
import drawning as drw
import os


# 0. Константы:
# 0.1. Изображение:
# фотография  шахматной доски, сделаная параллельно
IMG_FILE = "C:/work/miet/mag_sem_3/computer_vision/lab1/input/chessboard/24_cm.jpg"
# размер одной клетки в миллиметрах по [вертикали, горизонтали]
SQUARE_SIZES_MM = [30, 30]
# Общее количество пересечений клеток (ключевых точек)
OBJNUM = 86

# 0.2. Условия съемки:
# расстояние между изображением и камерой в мм
DISTANCE_MM = 24000

# 0.3. Параметры камеры:
# размеры пикселя по [вертикали, горизонтали]
PIXEL_SIZES_MM = [0.8 * 10 ** -6, 0.8 * 10 ** -6]

# 0.4. Настройки для программного кода:
# размер апертуры дифференциального оператора
IMPOC_HSIZE = 24
# допустимая область отклонения точки относительно ее предполагаемого положения (в пикс) [y, x]
AREA_OF_OBJ = [40, 40]


# 1. Открываем изображение:
imgi = np.uint8(cv2.imread(IMG_FILE, cv2.IMREAD_GRAYSCALE))
drw.show_image(imgi, title="base image")

# 2. Выделяем углы клеток на изображении:
img0 = improc.corners_highlighting(imgi, IMPOC_HSIZE)
drw.show_image(img0, title="image after differential operator")

# 3. Формируем из найденных углов объекты:
objs = improc.corners_to_objs(img0.copy(), OBJNUM, IMPOC_HSIZE)
drw.show_objects(
    objs, "+", "red", title="image after differential operator with found objects"
)

# 4. Строим предполагаемую сетку по найденным объектам:
objs_ideal = optics.objs_ideal(objs, AREA_OF_OBJ)
drw.show_objects(
    objs_ideal, "o", "blue", title='red - found objects\nblue - "ideal" objects'
)

# 5. Фильтруем найденные и идеальные объекты по их совпадению:
objs, objs_ideal = optics.objs_matching(objs, objs_ideal, AREA_OF_OBJ)
drw.show_objects(objs, "+", "red", img=img0, title="matching objects", waituser=False)
drw.show_objects(objs_ideal, "o", "blue", title="matching objects: %d" % len(objs))

# 6. Расчет фокусного расстояния:
focus_mm = optics.focus_calc(
    objs, imgi.shape, PIXEL_SIZES_MM, DISTANCE_MM, SQUARE_SIZES_MM, AREA_OF_OBJ
)
print("Focus (mm): %0.2f (y), %0.2f (x)" % (focus_mm[0], focus_mm[1]))
# 7. Расчитываем угол зрения:

FOV = optics.FOV_calc(imgi.shape, PIXEL_SIZES_MM, focus_mm)
print("Field of view (°): %d (y), %d (x)" % (FOV[0], FOV[1]))

# 8. Расчет коэффициентов радиальной дисторсии:
dist_coef, tstpoints = optics.dist_coeff(objs, objs_ideal, imgi.shape)
drw.show_objects_circules(
    tstpoints,
    "blue",
    img=np.array([[0] * imgi.shape[1]] * imgi.shape[0]),
    title="distortion",
)

# 9. Корректировка дисторсии на изображении:
imgo = optics.dist_img_correction(imgi, dist_coef)
drw.show_image(
    imgo,
    title="Number of found points: %d\nFocus (mm): %0.2f (y), %0.2f (x)\nField of view (°): %d (y), %d (x)"
    % (len(objs), focus_mm[0], focus_mm[1], FOV[0], FOV[1]),
)

# 10. Сохраняем результаты в файл:
camera = {
    "PIXEL_SIZES_MM": PIXEL_SIZES_MM,
    "FOCUSES_MM": focus_mm,
    "FOVs": FOV,
    "DIST_COEF": dist_coef,
}
file = open(os.path.join(os.path.dirname(__file__), "../../../provided/camera.txt"), "w")
file.write("%s = %s\n" % ("camera", camera))
file.close()
