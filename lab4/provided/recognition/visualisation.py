import numpy as np

def img_concat(berrys):
    x = 0
    img = np.uint8(np.ones((30, 180 * 4, 3)) * 255)
    for berry in berrys:

        if x == 0:
            imgl = berry.img
        else:
            imgl = np.hstack((imgl, berry.img))
        x = x + 1
        if x == 4:
            x = 0
            img = np.vstack((img, imgl))

    return img


def features_separate(berrys):

    black_currant = []
    blackberry = []
    blueberry = []
    raspberry = []

    for berry in berrys:

        # 2.1.1. convert features to arrays:
        if berry.group == "black_currant":
            black_currant.append([berry.feauters.size, berry.feauters.color_gray])
        if berry.group == "blackberry":
            blackberry.append([berry.feauters.size, berry.feauters.color_gray])
        if berry.group == "blueberry":
            blueberry.append([berry.feauters.size, berry.feauters.color_gray])
        if berry.group == "raspberry":
            raspberry.append([berry.feauters.size, berry.feauters.color_gray])

    black_currant = np.array(black_currant)
    blackberry = np.array(blackberry)
    blueberry = np.array(blueberry)
    raspberry = np.array(raspberry)

    return black_currant, blackberry, blueberry, raspberry


def all_image_plot(ax, img):
    ax.clear()
    ax.title.set_text("Изображения ягод")
    ax.text(180 * 0 + 20, 22, "black_currant")
    ax.text(180 * 1 + 20, 22, "blackberry")
    ax.text(180 * 2 + 20, 22, "blueberry")
    ax.text(180 * 3 + 20, 22, "raspberry")
    ax.imshow(img)
    ax.set_xlim([0, 180 * 4])
    ax.set_ylim([30 + 180 * 5, 0])
    for i in range(0, 5):
        ax.plot([0, 180 * 4], [30 + 180 * i, 30 + 180 * i], linewidth=1, c='k')
    for i in range(1, 4):
        ax.plot([180 * i, 180 * i], [0, 30 + 180 * 5], linewidth=1, c='k')

def berry_image_view(ax, berry):
    ax.clear()
    ax.imshow(berry.img)
    ax.set_title(berry.filename)

def plot_feautres(ax, black_currant, blackberry, blueberry, raspberry):
    ax.clear()
    ax.title.set_text("Пространство признаков")
    ax.grid()
    ax.scatter(black_currant[:, 0], black_currant[:, 1], label="black_currant", c="black")
    ax.scatter(blackberry[:, 0], blackberry[:, 1], label="blackberry", c="gray")
    ax.scatter(blueberry[:, 0], blueberry[:, 1], label="blueberry", c="blue")
    ax.scatter(raspberry[:, 0], raspberry[:, 1], label="raspberry", c="red")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("размер, нормир.")
    ax.set_ylabel("яркость, нормир.")


def show_and_wait_any_key(fig):
    fig.show()
    while fig.waitforbuttonpress() == False:
        pass