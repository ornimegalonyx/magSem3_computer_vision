import numpy as np
import matplotlib.pyplot as plt
import visualisation as vis
from berry import Berry
from rule import Rule
from perceptron import Perceptron

# 0. Parameters:
DATASET_PATH = "C:/work/miet/mag_sem_3/computer_vision/lab4/provided/data/dataset/"
DATA_PATH = "C:/work/miet/mag_sem_3/computer_vision/lab4/provided/data/"


# 1. Init dataset:
dataset = {}
for i in range(1, 6):
    dataset["black_currant_%d.png" % i] = "black_currant"
    dataset["blackberry_%d.png" % i] = "blackberry"
    dataset["blueberry_%d.png" % i] = "blueberry"
    dataset["raspberry_%d.png" % i] = "raspberry"


# 2. Get objects and its features:
berrys = []
for filename in dataset:
    berrys.append(Berry(DATASET_PATH + filename, group=dataset[filename]))


# 3. Visualizing:

## 3.1. Getting data from berrys:
img_all = vis.img_concat(berrys)
black_currant, blackberry, blueberry, raspberry = vis.features_separate(berrys)

## 3.2. Plotting data:
fig, ax = plt.subplots(1, 2)
fig.suptitle("Входные объекты и их признаки (нажмите пробел для продолжения)")
vis.all_image_plot(ax[0], img_all)
vis.plot_feautres(ax[1], black_currant, blackberry, blueberry, raspberry)
vis.show_and_wait_any_key(fig)


# 4. Rule for binary classification ((blueberry and blackberry)/not (blueberry and blackberry)):

## 4.1. points for blueberry and blackberry field:
points = np.array([[0.8, 0.33], [0.60, 0.275], [0.25, 0.55], [0.3, 0.6]])
rule = Rule(points)

## 4.2. visualisation:
fig.suptitle(
    "Область признаков для blueberry и blackberry (нажмите пробел для продолжения)"
)
ax[1].plot(points[:, 0], points[:, 1], "*b")
ax[1].plot(points[:, 0], points[:, 1], "b")
ax[1].plot(points[[-1, 0], 0], points[[-1, 0], 1], "b")
ax[1].fill(points[:, 0], points[:, 1], alpha=0.1, color="b")
vis.show_and_wait_any_key(fig)


# 5. Testing:
print("Binary classification with classic method:")
fig.suptitle(
    "Распознавание черники (blueberry и blackberry) по изображению (нажмите пробел для продолжения)"
)
for i in range(1, 5):

    ## 5.1. get unknown berry:
    berry = Berry(DATA_PATH + "berry_%d.png" % i)
    features = [berry.feauters.size, berry.feauters.color_gray]

    ## 5.2. visualisation:
    vis.berry_image_view(ax[0], berry)
    (plt_p,) = ax[1].plot(features[0], features[1], "+y", label="unknown")
    ax[1].legend()

    ## 5.3. recognition:
    flags = rule.check(features)
    group = ("not blueberry or blackberry", "r")
    if flags[0] & flags[1] & ~flags[2] & ~flags[3]:
        group = ("blueberry or blackberry", "g")

    ## 5.4. pass recognition result to output:
    ax[0].text(10, 10, "%s" % group[0], backgroundcolor=group[1])
    print(" %s is %s" % (berry.filename, group[0]))

    ## 5.5 wait user action in form:
    vis.show_and_wait_any_key(fig)
    plt_p.remove()


# 6. Perceptron for binary classification ((raspberry and blackberry)/not (raspberry and blackberry)):

## 6.1. get features and answers for trainig:
features = []
answers = []
for berry in berrys:
    features.append([berry.feauters.size, berry.feauters.color_gray])
    if (berry.group == "raspberry") or berry.group == "blackberry":
        answers.append([1])
    else:
        answers.append([0])

## 6.2. creating and training perceptron:
berry_nn = Perceptron(features, answers)

## 6.3. getting field of blueberry:
mask = []
for x in np.arange(-0.1, 1.1, 0.01):
    for y in np.arange(-0.1, 1.1, 0.01):
        mask.append([x, y])
mask = np.array(mask)
mask_ans = berry_nn.check(mask)[:, 0]

## 6.4. Normalizing features of berrys:
black_currant = berry_nn.data_normalization(black_currant)
blackberry = berry_nn.data_normalization(blackberry)
blueberry = berry_nn.data_normalization(blueberry)
raspberry = berry_nn.data_normalization(raspberry)

## 6.5. visualizing:
vis.all_image_plot(ax[0], img_all)
vis.plot_feautres(ax[1], black_currant, blackberry, blueberry, raspberry)
ax[1].scatter(mask[mask_ans, 0], mask[mask_ans, 1], s=2, alpha=0.2, c="blue")
ax[1].scatter(mask[~mask_ans, 0], mask[~mask_ans, 1], s=2, alpha=0.2, c="gray")
ax[1].set_xlim([-0.1, 1.1])
ax[1].set_ylim([-0.1, 1.1])
vis.show_and_wait_any_key(fig)


# 7. Testing perceptron:
print("Binary classification with perceptron:")
fig.suptitle(
    "Распознавание черники (blueberry и blackberry) по изображению (нажмите пробел для продолжения)"
)
for i in range(1, 5):

    ## 7.1. get unknown berry:
    berry = Berry(DATA_PATH + "berry_%d.png" % i)
    features = berry_nn.data_normalization(
        [berry.feauters.size, berry.feauters.color_gray]
    )

    ## 7.2. visualisation:
    vis.berry_image_view(ax[0], berry)
    (plt_p,) = ax[1].plot(features[0], features[1], "+y", label="unknown")
    ax[1].legend()

    ## 7.3. recognition:
    ans = berry_nn.check(features)[0]
    group = ("not raspberry or blackberry", "r")  ################################
    if ans:
        group = ("raspberry or blackberry", "g")  ################################

    ## 7.4. pass recognition result to output:
    ax[0].text(10, 10, "%s" % group[0], backgroundcolor=group[1])
    print(" %s is %s" % (berry.filename, group[0]))

    ## 7.5 wait user action in form:
    vis.show_and_wait_any_key(fig)
    plt_p.remove()
