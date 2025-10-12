from typing import List
import time

import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

from robo_algo.arm import RoboticArmPlotter
from robo_algo.constants import *
import robo_algo.core as core
from robo_algo.core import Color


DRAWING1: List[np.ndarray] = [
    np.array([
        [12, 2],
        [5, 5],
        [2, 5],
        [-3, 7],
        [-1, 4.5],
        [-5, 4],
        [-11, -1],
        [-7, -5],
        [-1, -7],
        [3, -8],
        [5, -11],
        [4, -7],
        [6, -5],
        [3, -6],
        [2, -6],
        [-1, -5],
        [-6, -2],
        [0, -1],
        [-1, -3],
        [2, -2],
        [3, -1],
        [5, -1],
        [4, 0],
        [8, 0],
        [9, 1],
        [12, 2],
        [7.5, 3],
    ]),
]
for i in range(len(DRAWING1)):
    DRAWING1[i] = (DRAWING1[i].astype(float) + 12.0) / 2.0


DRAWING3: List[np.ndarray] = [
    np.array([
        [-6, 11],
        [-3, 11],
        [-3, 10],
        [-4, 10],
        [-4, 6],
        [-5, 6],
        [-5, 10],
        [-6, 10],
        [-6, 11],
    ]),
    np.array([
        [-2, 11],
        [-1, 11],
        [-1, 9],
        [0, 9],
        [0, 11],
        [1, 11],
        [1, 6],
        [0, 6],
        [0, 8],
        [-1, 8],
        [-1, 6],
        [-2, 6],
        [-2, 11]
    ]),
    np.array([
        [2, 11],
        [5, 11],
        [5, 10],
        [3, 10],
        [3, 9],
        [5, 9],
        [5, 8],
        [3, 8],
        [3, 7],
        [5, 7],
        [5, 6],
        [2, 6],
        [2, 11],
    ]),
    np.array([
        [-21, 9],
        [-15, 9],
        [-14, 8],
        [-13, 6],
        [-13, 4],
        [-15, 2],
        [-13, -1],
        [-13, -4],
        [-14, -6],
        [-15, -7],
        [-21, -7],
        [-20, -6],
        [-20, 8],
        [-21, 9],
    ]),
    np.array([
        [-18, 3],
        [-18, 8],
        [-17, 8],
        [-16, 7],
        [-16, 4],
        [-17, 3],
        [-18, 3],
    ]),
    np.array([
        [-18, 1],
        [-17, 1],
        [-16, 0],
        [-16, -4],
        [-17, -5],
        [-18, -5],
        [-18, 1],
    ]),
    np.array([
        [-13, 3],
        [-8, 3],
        [-8, 1],
        [-9, 2],
        [-10, 2],
        [-10, -1],
        [-9, -1],
        [-8, 0],
        [-8, -3],
        [-9, -2],
        [-10, -2],
        [-10, -6],
        [-9, -6],
        [-8, -5],
        [-8, -7],
        [-13, -7],
        [-12, -6],
        [-12, 2],
        [-13, 3],
    ]),
    np.array([
        [-8, -7],
        [-7, -6],
        [-5, 2],
        [-5, 3],
        [-3, 3],
        [-3, 2],
        [-1, -6],
        [0, -7],
        [-4, -7],
        [-3, -6],
        [-3, -4],
        [-5, -4],
        [-5, -6],
        [-4, -7],
        [-8, -7],
    ]),
    np.array([
        [-5, -3],
        [-3, -3],
        [-4, 0],
    ]),
    np.array([
        [-2, 1],
        [-2, 3],
        [4, 3],
        [4, 1],
        [3, 2],
        [2, 2],
        [2, -9],
        [3, -10],
        [-1, -10],
        [0, -9],
        [0, 2],
        [-1, 2],
        [-2, 1],
    ]),
    np.array([
        [4, 3],
        [8, 3],
        [7, 2],
        [7, -6],
        [8, -6],
        [9, -5],
        [9, -7],
        [4, -7],
        [5, -6],
        [5, 2],
        [4, 3],
    ]),
    np.array([
        [9, 3],
        [14, 3],
        [14, 1],
        [13, 2],
        [12, 2],
        [12, -1],
        [13, -1],
        [14, 0],
        [14, -3],
        [13, -2],
        [12, -2],
        [12, -6],
        [13, -6],
        [14, -5],
        [14, -7],
        [9, -7],
        [10, -6],
        [10, 2],
        [9, 3],
    ]),
    np.array([
        [15, 1],
        [17, 3],
        [19, 3],
        [21, 2],
        [20, 1],
        [19, 2],
        [18, 2],
        [17, 1],
        [17, 0],
        [18, -1],
        [20, -2],
        [21, -3],
        [21, -5],
        [19, -7],
        [15, -6],
        [15, -5],
        [16, -4],
        [16, -5],
        [17, -6],
        [18, -6],
        [19, -5],
        [19, -4],
        [18, -3],
        [16, -2],
        [15, -1],
        [15, 1],
    ])
]
for i in range(len(DRAWING3)):
    DRAWING3[i] = (DRAWING3[i].astype(float) + 22.0) / 3.0


def get_drawing1():
    return DRAWING1


def get_drawing2():
    def stack(t, f):
        return np.stack((t, f)).T

    t = np.arange(start=0, stop=20, step=0.5)
    sint = np.sin(t) + 5
    cost = np.cos(t) + 5
    return [stack(t, sint),
            stack(t, cost)]


def get_drawing3():
    return DRAWING3


if __name__ == "__main__":
    ctx = core.RenderingContext("graph test")
    arm1 = RoboticArmPlotter(
        ctx,
        joint0_position=np.array([8, 8]),
        link_lengths=[2, 3, 4],
        link_angles=[np.deg2rad(0), np.deg2rad(10), np.deg2rad(0)],
        thickness=0.1,
        color=Color(127, 127, 127, 255),
        joint_radius=0.3,
        joint_color=Color(200, 200, 200, 255),
    )
    arm1.start_drawing()

    running = True
    spf_running_mean = 0
    coef = 0
    i_point = 0 
    i_shape = 0
    drawing1 = get_drawing1()
    drawing2 = get_drawing2()
    drawing3 = get_drawing3()
    try:
        while running:
            # Check the event queue
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    # The user closed the window or pressed escape
                    running = False

            ms_start = time.perf_counter_ns() / 1000
            ctx.screen.fill((0, 0, 0, 0))

            drawing = drawing3  # drawing1, drawing2, drawing3
            if i_shape < len(drawing):
                pt = drawing[i_shape][i_point]
                arm1.draw(pt)
                if (i_point + 1) >= len(drawing[i_shape]):
                    i_shape += 1
                    i_point = 0
                    arm1.stop_drawing()
                    arm1.start_drawing()
                else:
                    i_point += 1

            arm1.render()

            # Make Box2D simulate the physics of our world for one step.
            ctx.world.Step(TIME_STEP, 10, 10)
            
            spf = time.perf_counter_ns() / 1000 - ms_start
            spf_running_mean = spf_running_mean * coef + (1 - coef) * spf
            coef = 0.99
            print(f"fps={1 / spf_running_mean * 1000 * 1000:.1f} [{spf_running_mean / 1000:.3f}ms]")

            # Flip the screen and try to keep at the target FPS
            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Terminating...")
    pygame.quit()