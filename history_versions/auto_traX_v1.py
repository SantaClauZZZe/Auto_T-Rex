# + Нажатие клавиш в отдельном потоке
# + Расчет фпс
# max score ~ 970

from time import perf_counter, sleep
import numpy as np

import cv2 as cv
from skimage.measure import label, regionprops

import pyautogui as pag

from mss import mss

from threading import Thread
from queue import Queue

import matplotlib.pyplot as plt

def calibrate():
    import winsound

    for i in range(4):
        winsound.Beep(440, 100)
        sleep(3)
        print(f"x = {pag.position().x}; y = {pag.position().y}")

    winsound.Beep(440, 100)
    return

def process_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    tresh = cv.threshold(gray, 130, 255, 1)[1]
    tresh = cv.dilate(tresh, None, iterations=2)
    tresh = cv.erode(tresh, None, iterations=4)

    cv.rectangle(img, (*obstacle_coords[0],), (*obstacle_coords[1],), (0, 0, 255), 2)

    return tresh

def process_key(task : Queue):
    while True:
        t = task.get()

        if t == None:
            break

        if t == 1:
            pag.keyDown('up')
        elif t == 2:
            pag.keyUp('up')
            pag.keyDown('down')
            pag.keyUp('down')

# ------------- main code -------------

# calibrate()

cv.namedWindow("Screen", cv.WINDOW_AUTOSIZE)

fps = 0
timer = perf_counter()

with mss() as screenshot:

    monitor = { "top" : 230,
                "left" : 540,
                "width" : 600,
                "height" : 100,
              }

    # dino_coords = (590 - monitor["left"], 310 - monitor["top"])
    obstacle_coords = ((630 - monitor["left"], 300 - monitor["top"]),
                       (660 - monitor["left"], 330 - monitor["top"]))

    is_jump = False
    tasks = Queue()
    thread = Thread(target=process_key, args=(tasks, ))
    thread.start()

    while True:
        if perf_counter() - timer >= 1.0:
            print(f"\nFPS: {fps}\n", end="")
            timer = perf_counter()
            fps = 0

        img = np.array(screenshot.grab(monitor))
        
        res_img = process_image(img)

        obstacle_mask = res_img[obstacle_coords[0][1]:obstacle_coords[1][1], obstacle_coords[0][0]:obstacle_coords[1][0]]

        if not is_jump and np.any(obstacle_mask):
            tasks.put(1)
            is_jump = True
        elif is_jump and not np.any(obstacle_mask):
            tasks.put(2)
            is_jump = False
        print(end="")
        fps += 1
        
        cv.imshow("Screen", img)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cv.destroyAllWindows()

tasks.put(None)
thread.join()
print("End program")