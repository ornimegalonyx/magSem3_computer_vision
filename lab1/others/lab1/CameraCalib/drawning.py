import numpy as np
import matplotlib.pyplot as plt

the_key = None

def press(event):
    global the_key
    the_key = event.key

def wait_user_keypress (plt, title, key):
    plt.gcf().canvas.mpl_connect('key_press_event', press)
    print(title + " was draws. Press 'n' to continue")
    global the_key
    while True:
        plt.waitforbuttonpress()
        if the_key == key:
            the_key = None
            break
    print("Continue...")

def show_image(img, title="", waituser=True):
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=np.min(0), vmax=np.max(img))
    plt.title(title)
    plt.show(block=False)
    if waituser:
        wait_user_keypress(plt, title, 'n')

def show_objects(objs, sym, color, img=None, title="", waituser=True):
    if img is not None:
        show_image(img, title=title, waituser=False)
    else:
        plt.title(title)
    for obj in objs:
        plt.text(obj[1], obj[0], sym, color=color)
    if waituser:
        wait_user_keypress(plt, title, 'n')

def show_objects_circules(objs, color, img=None, title="", waituser=True):
    if img is not None:
        show_image(img, title=title, waituser=False)
    else:
        plt.title(title)
    for i in range(len(objs)):
        plt.gcf().gca().add_artist(plt.Circle((objs[i][1], objs[i][0]), 3, color="blue"))
    if waituser:
        wait_user_keypress(plt, title, 'n')
