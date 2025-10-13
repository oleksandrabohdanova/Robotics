from typing import List
import time

import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

from robo_algo import arm_controller
from robo_algo.arm import RoboticArm, RoboticArmPlotter
from robo_algo.arm_controller import ArmController
from robo_algo.plotter_graphs import get_drawing1
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

    # Cumulative angles
    cumulative_angles = np.cumsum(link_angles)

    # Calculate derivatives for each angle
    for i in range(n):
        for k in range(i, n):
            jacobian_matrix[0, i] += -link_lengths[k] * np.sin(cumulative_angles[k])
            jacobian_matrix[1, i] += link_lengths[k] * np.cos(cumulative_angles[k])

    return jacobian_matrix


def inverse_kinematics(target_position, arm, max_iterations=200, tolerance=0.001):
    # Starting angles
    angles = arm.get_angles().copy()
    link_lengths = np.array(arm.link_lengths)
    
    # Nullspace projection parameters
    k_base = 0.02  # Base gain for secondary task
    theta_ref = np.zeros(len(angles))  # Reference configuration
    limit_threshold = 2.5  # Start caring about limits beyond this value

    # Iterative optimization
    for iteration in range(max_iterations):
        # Current end effector position
        current_position = forward_kinematics(arm, angles)

        # Error between desired and current position
        error = target_position - current_position
        error_magnitude = np.linalg.norm(error)

        # If close enough - exit
        if error_magnitude < tolerance:
            break

        J = jacobian(link_lengths, angles)

        J_pinv = np.linalg.pinv(J)

        delta_theta_primary = J_pinv @ error
        
        delta_theta_secondary = np.zeros(len(angles))
        
        # Check if any joint is approaching limits
        joints_near_limits = np.abs(angles) > limit_threshold
        
        if np.any(joints_near_limits) and error_magnitude < 0.5:
            # Calculate adaptive gain based on error 
            k_adaptive = k_base * min(1.0, error_magnitude / 0.1)
            
            # Nullspace projector
            I = np.eye(len(angles))
            N = I - J_pinv @ J
            
            # Only push joints that are near limits
            limit_gradient = np.zeros(len(angles))
            for i in range(len(angles)):
                if np.abs(angles[i]) > limit_threshold:
                    # Push away from limit, not just towards center
                    limit_gradient[i] = -np.sign(angles[i]) * (np.abs(angles[i]) - limit_threshold)
            
            delta_theta_secondary = k_adaptive * N @ limit_gradient
        
        delta_angles = delta_theta_primary + delta_theta_secondary

        # Update angles
        angles = angles + delta_angles

    return angles


#######################################################################################
#######################################################################################


if __name__ == "__main__":
    ctx = core.RenderingContext("Task 1 - visualization")
    arm = RoboticArmPlotter(
        ctx,
        joint0_position=np.array([8, 8]),
        link_lengths=[5, 4, 3],
        link_angles=[np.deg2rad(-20), np.deg2rad(140), np.deg2rad(180)],
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
    drawing1: List[np.ndarray] = get_drawing1()

    running = True
    spf_running_mean = 0
    coef = 0
    i_shape = 0
    i_point = 0
    moving_to_start = False
    try:
        while running:
            # Check the event queue
            for event in pygame.event.get():
                if event.type == QUIT or (
                    event.type == KEYDOWN and event.key == K_ESCAPE
                ):
                    running = False

            ms_start = time.perf_counter_ns() / 1000
            ctx.screen.fill((0, 0, 0, 0))
            arm.render()
            
            ############################## YOUR CODE GOES HERE ####################################
            #######################################################################################
            # UA: Код для обчислення зворотної кінематики та малювання має бути тут               #
            # EN: Your code for IK and drawing goes here.                                         #
            #######################################################################################
            
            drawing = drawing1

            if i_shape < len(drawing):
                current_shape = drawing[i_shape]

                if controller.is_idle():
                    if i_point < len(current_shape):
                        # If this is first point of shape - move without drawing
                        if i_point == 1 and moving_to_start:
                            arm.start_drawing()
                            moving_to_start = False

                        if i_point == 0:
                            moving_to_start = True

                        # Get next point
                        target_point = current_shape[i_point]

                        # IK to calculate angles
                        target_angles = inverse_kinematics(target_point, arm)

                        # Move arm
                        controller.move_to_angles(target_angles)

                        i_point += 1
                    else:
                        # Shape finished
                        arm.stop_drawing()
                        i_shape += 1
                        i_point = 0
                else:
                    # Moving
                    controller.step()

                    if not moving_to_start:
                        arm.draw()

            #######################################################################################
            #######################################################################################

            # Make Box2D simulate the physics 
            ctx.world.Step(TIME_STEP, 10, 10)
            spf = time.perf_counter_ns() / 1000 - ms_start
            spf_running_mean = spf_running_mean * coef + (1 - coef) * spf
            coef = 0.99
            print(
                f"fps={1 / spf_running_mean * 1000 * 1000:.1f} [{spf_running_mean / 1000:.3f}ms]"
            )
           
            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Terminating...")
    pygame.quit()