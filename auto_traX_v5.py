# + Дино теперь умеет наклонятся
# max score 63316
# 
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

    cv.rectangle(img, (*obstacle_coords_down[0],), (*obstacle_coords_down[1],), (255, 0, 0), 2)

    cv.rectangle(img, (*safely_coords[0], ), (*safely_coords[1], ), (0, 255, 0), 2)

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
    is_down = False

    while True:
        t = task.get()

        if t == None:
            break

        if t[0] == -1:
            if is_down:
                pag.keyUp('down')
                is_down = False
            else:
                pag.keyUp('up')
                pag.keyDown('down')
                pag.keyUp('down')
        elif t[0] == 1:
            pag.keyDown('up')
        else:
            pag.keyDown('down')
            sleep(t[1])
            is_down = True

# ------------- main code -------------

# calibrate()

cv.namedWindow("Screen", cv.WINDOW_AUTOSIZE)

fps = 0
last_fps = 0
timer = perf_counter()

with mss() as screenshot:

    monitor = { "top" : 230,
                "left" : 540,
                "width" : 600,
                "height" : 100,
              }

    # dino_x = 590 - monitor["left"]
    # delta_x = 40

    # base_obstacle_coords = ((dino_x + delta_x, 300 - monitor["top"]),
    #                         (dino_x + delta_x + 20, 330 - monitor["top"]))

    base_obstacle_coords = ((630 - monitor["left"], 300 - monitor["top"]),
                            (650 - monitor["left"], 330 - monitor["top"]))

    obstacle_coords = ((base_obstacle_coords[0][0], base_obstacle_coords[0][1]), (base_obstacle_coords[1][0], 85))
    obstacle_coords_down = ((base_obstacle_coords[0][0], 85), (base_obstacle_coords[1][0], base_obstacle_coords[1][1]))

    safely_coords = base_obstacle_coords

    width_window = 20

    is_jump = False
    tasks = Queue()
    thread = Thread(target=process_key, args=(tasks, ))
    thread.start()
    min_speed = 6
    _speed = 6
    speed_timer = perf_counter()

    while True:
        if perf_counter() - timer >= 1.0:
            print(f"\rFPS: {fps} ; Speed: {_speed}", end="")
            timer = perf_counter()
            last_fps = fps
            fps = 0

        img = np.array(screenshot.grab(monitor))
        
        res_img = process_image(img)

        safely_mask = res_img[safely_coords[0][1] : safely_coords[1][1], safely_coords[0][0] : safely_coords[1][0]]
        obstacle_mask = res_img[obstacle_coords[0][1] : obstacle_coords[1][1], obstacle_coords[0][0] : obstacle_coords[1][0]]
        obstacle_mask_down = res_img[obstacle_coords_down[0][1] : obstacle_coords_down[1][1], obstacle_coords_down[0][0] : obstacle_coords_down[1][0]]

        temp = calc_speed(res_img)
        if temp and temp - _speed < 3 and temp >= min_speed:
            _speed = temp
            if perf_counter() -  speed_timer > 15:
                speed_timer = perf_counter()
                k = float(_speed / 6)
                x0 = int(base_obstacle_coords[0][0] * k)
                obstacle_coords = ((x0, obstacle_coords[0][1]), (x0 + width_window, obstacle_coords[1][1]))
                obstacle_coords_down = ((x0, obstacle_coords_down[0][1]), (x0 + width_window, obstacle_coords_down[1][1]))
                safely_coords = ((x0 - width_window, base_obstacle_coords[0][1]), (x0 + width_window, base_obstacle_coords[1][1]))

        # print(f"\rSpeed: {_speed}", end="")

        if not is_jump:
            obstacle_up = np.any(obstacle_mask)
            obstacle_down = np.any(obstacle_mask_down)

            if obstacle_up or obstacle_down:
                if obstacle_down:
                    tasks.put((1, None))
                else:
                    tasks.put((2, obstacle_coords[1][0] / _speed / last_fps))
                is_jump = True
        elif is_jump and not np.any(safely_mask):
            tasks.put((-1, None))
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
print("\nEnd program")