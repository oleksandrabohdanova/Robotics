from typing import List
import time

import numpy as np
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

from robo_algo.arm import RoboticArm, RoboticArmPlotter
from robo_algo.arm_controller import ArmController
from robo_algo.plotter_graphs import get_drawing3
from robo_algo.constants import *
import robo_algo.core as core
from robo_algo.core import Color


MAX_SPEED = np.deg2rad(1.0)

############################## YOUR CODE GOES HERE ####################################
#######################################################################################
# UA: Код для обчислення кінематики має бути тут               #
# EN: Your code for IK and drawing goes here.                                         #

def forward_kinematics(arm: RoboticArm, angles=None):
    base_position = np.array(arm.joints[0].position)
    link_lengths = np.array(arm.link_lengths)
    if angles is None:
        angles = np.array(arm.get_angles())

    cumulative_angles = np.cumsum(angles)
    link_vectors = np.stack(
        [
            link_lengths * np.cos(cumulative_angles),
            link_lengths * np.sin(cumulative_angles),
        ],
        axis=1,
    )
    end_effector_position = base_position + np.sum(link_vectors, axis=0)
    return end_effector_position[:2]

def jacobian(link_lengths, link_angles):
    n = len(link_angles)
    jacobian_matrix = np.zeros((2, n))
    cumulative_angles = np.cumsum(link_angles)
    for i in range(n):
        for k in range(i, n):
            jacobian_matrix[0, i] += -link_lengths[k] * np.sin(cumulative_angles[k])
            jacobian_matrix[1, i] += link_lengths[k] * np.cos(cumulative_angles[k])
    return jacobian_matrix

def inverse_kinematics(target_position, arm, max_iterations=300, tolerance=0.0005):
   
    angles = arm.get_angles().copy()
    link_lengths = np.array(arm.link_lengths)
    k_base = 0.01  
    limit_threshold = 2.5

    for iteration in range(max_iterations):
        current_position = forward_kinematics(arm, angles)
        error = target_position - current_position
        error_magnitude = np.linalg.norm(error)

        if error_magnitude < tolerance:
            break

        J = jacobian(link_lengths, angles)
        J_pinv = np.linalg.pinv(J)
        delta_theta_primary = J_pinv @ error
        delta_theta_secondary = np.zeros(len(angles))
        
        joints_near_limits = np.abs(angles) > limit_threshold
        if np.any(joints_near_limits) and error_magnitude < 0.5:
            k_adaptive = k_base * min(1.0, error_magnitude / 0.1)
            I = np.eye(len(angles))
            N = I - J_pinv @ J
            limit_gradient = np.zeros(len(angles))
            for i in range(len(angles)):
                if np.abs(angles[i]) > limit_threshold:
                    limit_gradient[i] = -np.sign(angles[i]) * (np.abs(angles[i]) - limit_threshold)
            delta_theta_secondary = k_adaptive * N @ limit_gradient
        
        delta_angles = delta_theta_primary + delta_theta_secondary
        angles = angles + delta_angles

    return angles

#######################################################################################
#######################################################################################


if __name__ == "__main__":
    ctx = core.RenderingContext("Task 3 - visualization")
    arm = RoboticArmPlotter(
        ctx,
        joint0_position=np.array([8, 8]),
        link_lengths=[5, 4, 5, 3, 4, 2],
        link_angles=[np.deg2rad(45), np.deg2rad(100), np.deg2rad(-150), np.deg2rad(-129), np.deg2rad(-64), np.deg2rad(-300)],
        thickness=0.1,
        color=Color(127, 127, 127, 255),
        joint_radius=0.3,
        joint_color=Color(200, 200, 200, 255),
    )
    controller = ArmController(arm, max_velocity=MAX_SPEED)

    ### <UA>
    ### drawing1 це список масивів точок. Кожен масив описує фігуру для малювання.
    ### Кожна фігура малюється окремо. Не повинно бути ліній, які зʼєднують дві фігури.
    ### Кожна фігура це масив точок, які повинні бути намальовані в прямому порядку.
    ### <EN>
    ### drawing1 is a list of arrays of points. Each array describes a shape to draw.
    ### Each shape must be plotted separately. No line must connect two shapes.
    ### Each shape is an array of points which must be plotted in order.
    drawing3: List[np.ndarray] = get_drawing3()

    running = True
    i_shape = 0
    i_point = 0
    moving_to_start = False
    try:
        while running:
            # Check the event queue
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                   
                    running = False

            ctx.screen.fill((0, 0, 0, 0))
            arm.render()

            ############################## YOUR CODE GOES HERE ####################################
            #######################################################################################
            # UA: Код для обчислення зворотної кінематики та малювання має бути тут               #
            # EN: Your code for IK and drawing goes here.                                         #
            #######################################################################################
            #######################################################################################
           
            drawing = drawing3

            if i_shape < len(drawing):
                current_shape = drawing[i_shape]
                
                if controller.is_idle():
                    if i_point < len(current_shape):
                        target_point = current_shape[i_point]
                        
                        # Check if target is reachable
                        max_reach = np.sum(np.array(arm.link_lengths))
                        distance_from_base = np.linalg.norm(target_point - arm.joints[0].position)
                        
                        if distance_from_base > max_reach * 0.95:
                            # Skip unreachable points
                            i_point += 1
                            if i_point >= len(current_shape):
                                arm.stop_drawing()
                                i_point = 0
                                i_shape += 1
                            continue
                        
                        # Starting new shape - move without drawing
                        if i_point == 0:
                            arm.stop_drawing()
                            moving_to_start = True
                        
                        # After reaching first point - start drawing
                        if i_point == 1 and moving_to_start:
                            arm.start_drawing()
                            moving_to_start = False
                        
                        target_angles = inverse_kinematics(target_point, arm)
                        controller.move_to_angles(target_angles)
                        i_point += 1
                    else:
                        # Finished shape - stop drawing before next
                        arm.stop_drawing()
                        i_point = 0
                        i_shape += 1
                else:
                    controller.step()
                    # Only draw when not moving to start of new shape
                    if not moving_to_start:
                        arm.draw()

            #######################################################################################
            #######################################################################################

            # Make Box2D simulate the physics 
            ctx.world.Step(TIME_STEP, 10, 10)
            
            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Terminating...")
    pygame.quit()