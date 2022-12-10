# + Расчет скорости
# max score ~ 1450

# https://chromedino.com/

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

    # cv.rectangle(img, (50, 30), (80, 40), (0, 255, 0), 2)

    return tresh

last_centroid = 0
last_area = 0
is_correct = False
speed = 0

def calc_speed(img):
    global last_centroid, last_area, is_correct, speed

    labeled = label(img)
    props = regionprops(labeled)

    if len(props) > 1:
        if props[1].centroid[1] > 150 and props[1].area == last_area:
            speed = np.abs(props[1].centroid[1] - last_centroid)
            last_centroid = props[1].centroid[1]
            if is_correct:
                return speed
            is_correct = True
        else:
            last_area = props[1].area
            is_correct = False
    else:
            is_correct = False

    return None

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
    base_obstacle_coords = ((630 - monitor["left"], 300 - monitor["top"]),
                       (660 - monitor["left"], 330 - monitor["top"]))
    obstacle_coords = base_obstacle_coords

    is_jump = False
    tasks = Queue()
    thread = Thread(target=process_key, args=(tasks, ))
    thread.start()
    _speed = 6
    speed_timer = perf_counter()

    while True:
        if perf_counter() - timer >= 1.0:
            print(f"\rFPS: {fps} ; Speed: {_speed}", end="")
            timer = perf_counter()
            fps = 0

        img = np.array(screenshot.grab(monitor))
        
        res_img = process_image(img)

        obstacle_mask = res_img[obstacle_coords[0][1]:obstacle_coords[1][1], obstacle_coords[0][0]:obstacle_coords[1][0]]

        temp = calc_speed(res_img)
        if temp and np.abs(temp - _speed) < 2:
            _speed = temp
            if perf_counter() -  speed_timer > 15:
                speed_timer = perf_counter()
                obstacle_coords = ((int(base_obstacle_coords[0][0] * float(_speed/6)), base_obstacle_coords[0][1]), (int(base_obstacle_coords[1][0] * float(_speed / 6)), base_obstacle_coords[1][1]))

        # print(f"\rSpeed: {_speed}", end="")

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