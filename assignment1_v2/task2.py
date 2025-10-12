from typing import List
import time

import numpy as np
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

from robo_algo.arm import RoboticArm, RoboticArmPlotter
from robo_algo.arm_controller import ArmController
from robo_algo.plotter_graphs import get_drawing2
from robo_algo.constants import *
import robo_algo.core as core
from robo_algo.core import Color


# Increase for debugging!
MAX_SPEED = np.deg2rad(1.0)

############################## YOUR CODE GOES HERE ####################################
#######################################################################################
# UA: Код для обчислення кінематики має бути тут               #
# EN: Your code for IK and drawing goes here.                                         #

def forward_kinematics(arm: RoboticArm, angles=None):
    return np.array([3., 3.])

def jacobian(link_lengths, link_angles):
    jacobian_matrix = np.zeros((2, len(link_angles)))
    return jacobian_matrix

def inverse_kinematics(target_position, arm):
    angles = arm.get_angles()
    return angles

#######################################################################################
#######################################################################################


if __name__ == "__main__":
    ctx = core.RenderingContext("Task 2 - visualization")
    arm = RoboticArmPlotter(
        ctx,
        joint0_position=np.array([8, 8]),
        link_lengths=[5, 4, 3, 2],
        link_angles=[np.deg2rad(160), np.deg2rad(100), np.deg2rad(-40), np.deg2rad(-180)],
        thickness=0.1,
        color=Color(127, 127, 127, 255),
        joint_radius=0.3,
        joint_color=Color(200, 200, 200, 255),
    )
    controller = ArmController(arm, max_velocity=MAX_SPEED)
    arm.start_drawing()

    ### <UA>
    ### drawing1 це список масивів точок. Кожен масив описує фігуру для малювання.
    ### Кожна фігура малюється окремо. Не повинно бути ліній, які зʼєднують дві фігури.
    ### Кожна фігура це масив точок, які повинні бути намальовані в прямому порядку.
    ### <EN>
    ### drawing1 is a list of arrays of points. Each array describes a shape to draw.
    ### Each shape must be plotted separately. No line must connect two shapes.
    ### Each shape is an array of points which must be plotted in order.
    drawing2: List[np.ndarray] = get_drawing2()

    running = True
    i_shape = 0
    i_point = 0
    try:
        while running:
            # Check the event queue
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    # The user closed the window or pressed escape
                    running = False

            ctx.screen.fill((0, 0, 0, 0))
            arm.render()

            ############################## YOUR CODE GOES HERE ####################################
            #######################################################################################
            # UA: Код для обчислення зворотної кінематики та малювання має бути тут               #
            # EN: Your code for IK and drawing goes here.                                         #
            #######################################################################################
            #######################################################################################
            # Hint: see arm_controller interface, use `controller`
            # controller.move_to_angles([your predicted angles])
            drawing = drawing2

            #######################################################################################
            #######################################################################################

            # Make Box2D simulate the physics of our world for one step.
            ctx.world.Step(TIME_STEP, 10, 10)
            # Flip the screen and try to keep at the target FPS
            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Terminating...")
    pygame.quit()