import matplotlib.pyplot as plt
import numpy as np


# Инициализация области для осей на фигуре:
def axes_init(rect, title, ticks="off"):
    ax = plt.axes(rect)  # left, bottom, width, height
    if ticks == "off":
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.title.set_text(title)

    return ax


# Обновление изображения на фигуре:
def img_update(ax, img, title):
    ax.clear()
    ax.imshow(img, cmap='gray', vmin=0, vmax=np.max(img))
    ax.title.set_text(title)


# Обновление гистограммы на фигуре:
def hist_update(ax_hist, img, title):
    ax_hist.clear()
    ax_hist.grid()
    ax_hist.hist(img.flatten(), range=(0, 255), bins=255)
    ax_hist.title.set_text(title)
