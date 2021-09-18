import cv2
import numpy as np
import image_processing as improc
import optics
import drawning as drw


# 0. Константы:
# 0.1. Изображение:
IMG_FILE        = "../data/chessboard/IMG_3769.JPG"    # фотография  шахматной доски, сделаная параллельно
SQUARE_SIZES_MM = [20, 20]                      # размер одной клетки в миллиметрах по [вертикали, горизонтали]
OBJNUM          = 86                            # Общее количество пересечений клеток (ключевых точек)

# 0.2. Условия съемки:
DISTANCE_MM     = 160                           # расстояние между изображением и камерой в мм

# 0.3. Параметры камеры:
PIXEL_SIZES_MM  = [1.7*10**-3, 1.7*10**-3]      # размеры пикселя по [вертикали, горизонтали]

# 0.4. Настройки для программного кода:
IMPOC_HSIZE     = 24                            # размер апертуры дифференциального оператора
AREA_OF_OBJ     = [40, 40]                      # допустимая область отклонения точки относительно ее предполагаемого положения (в пикс) [y, x]


# 1. Открываем изображение:
imgi = np.uint8(cv2.imread(IMG_FILE, cv2.IMREAD_GRAYSCALE))
drw.show_image(imgi, title="base image")

# 2. Выделяем углы клеток на изображении:
img0 = improc.corners_highlighting(imgi, IMPOC_HSIZE)
drw.show_image(img0, title="image after differential operator")

# 3. Формируем из найденных углов объекты:
objs = improc.corners_to_objs(img0.copy(), OBJNUM, IMPOC_HSIZE)
drw.show_objects(objs, '+', "red", title="image after differential operator with found objects")

# 4. Строим предполагаемую сетку по найденным объектам:
objs_ideal = optics.objs_ideal(objs, AREA_OF_OBJ)
drw.show_objects(objs_ideal, 'o', "blue", title="red - found objects\nblue - \"ideal\" objects")

# 5. Фильтруем найденные и идеальные объекты по их совпадению:
objs, objs_ideal = optics.objs_matching(objs, objs_ideal, AREA_OF_OBJ)
drw.show_objects(objs, '+', "red", img=img0, title="matching objects", waituser=False)
drw.show_objects(objs_ideal, 'o', "blue", title="matching objects: %d" % len(objs))

# 6. Расчет фокусного расстояния:
focus_mm = optics.focus_calc(objs, imgi.shape, PIXEL_SIZES_MM, DISTANCE_MM, SQUARE_SIZES_MM, AREA_OF_OBJ)
print ("Focus (mm): %0.2f (y), %0.2f (x)" % (focus_mm[0], focus_mm[1]))
# 7. Расчитываем угол зрения:

FOV = optics.FOV_calc(imgi.shape, PIXEL_SIZES_MM, focus_mm)
print ("Field of view (°): %d (y), %d (x)" % (FOV[0], FOV[1]))

# 8. Расчет коэффициентов радиальной дисторсии:
dist_coef, tstpoints = optics.dist_coeff(objs, objs_ideal, imgi.shape)
drw.show_objects_circules(tstpoints, "blue", img=np.array([[0] * imgi.shape[1]] * imgi.shape[0]), title="distortion")

# 9. Корректировка дисторсии на изображении:
imgo = optics.dist_img_correction(imgi, dist_coef)
drw.show_image(imgo, title="Number of found points: %d\nFocus (mm): %0.2f (y), %0.2f (x)\nField of view (°): %d (y), %d (x)"
                              % (len(objs), focus_mm[0], focus_mm[1], FOV[0], FOV[1]))

# 10. Сохраняем результаты в файл:
camera = {"PIXEL_SIZES_MM" : PIXEL_SIZES_MM, "FOCUSES_MM" : focus_mm, "FOVs" : FOV, "DIST_COEF": dist_coef}
file = open("../data/camera.txt", "w")
file.write("%s = %s\n" %("camera", camera))
file.close()
