from typing import List
import time

import numpy as np
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, MOUSEBUTTONDOWN)

from robo_algo.arm import RoboticArm, RoboticArmPlotter
from robo_algo.arm_controller import ArmController
from robo_algo.constants import *
import robo_algo.core as core
from robo_algo.core import Color, ColorGreen


MAX_SPEED = np.deg2rad(1.5)

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

def inverse_kinematics(target_position, arm, max_iterations=100, tolerance=0.01):
    
    # Get current angles
    angles = arm.get_angles().copy()
    link_lengths = np.array(arm.link_lengths)
    
    # Check if target is reachable
    base_pos = arm.joints[0].position
    distance_to_target = np.linalg.norm(target_position - base_pos)
    max_reach = np.sum(link_lengths)
    min_reach = 0.5  # Minimum distance from base
    
    # If unreachable - return current angles
    if distance_to_target > max_reach * 0.95 or distance_to_target < min_reach:
        return angles
    
    # Adaptive damping - increase near singularities
    base_damping = 0.01
    
    for i in range(max_iterations):
        # Current position
        current_position = forward_kinematics(arm, angles)
        
        error = target_position - current_position
        error_norm = np.linalg.norm(error)
        
        # Check if close enough
        if error_norm < tolerance:
            break
        
        J = jacobian(link_lengths, angles)
        
        damping = base_damping + 0.1 * min(1.0, error_norm / max_reach)
      
        J_pseudo_inverse = np.linalg.inv((J.T.dot(J)) + damping * np.eye(J.shape[1])).dot(J.T)
        
        # Update angles with smaller steps
        delta_angles = 0.5 * np.dot(J_pseudo_inverse, error)
        angles += delta_angles
    
    return angles

#######################################################################################
#######################################################################################


if __name__ == "__main__":
    ctx = core.RenderingContext("Task 4 - visualization")
    arm = RoboticArmPlotter(
        ctx,
        joint0_position=np.array([8, 8]),
        link_lengths=[2, 1, 1, 2, 1, 2],
        link_angles=[np.deg2rad(160), np.deg2rad(-80), np.deg2rad(130),
                     np.deg2rad(0), np.deg2rad(90), np.deg2rad(200)],
        thickness=0.1,
        color=Color(127, 127, 127, 255),
        joint_radius=0.3,
        joint_color=Color(200, 200, 200, 255),
    )
    controller = ArmController(arm, max_velocity=MAX_SPEED)
    arm.start_drawing()

    pygame.font.init() 
                  
    my_font = pygame.font.SysFont(pygame.font.get_default_font(), 30)

    running = True
    target_point = np.array([5., 5.])
    start = 0
    try:
        while running:
            # Check the event queue
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    target_point = core.from_pix(np.array(event.dict['pos']))
                elif event.type == MOUSEBUTTONDOWN:
                    target_point = core.from_pix(np.array(event.dict['pos']))

            ctx.screen.fill((0, 0, 0, 0))

            text_surface = my_font.render(
                f"Target: {target_point[0]:.2f}, {target_point[1]:.2f}",
                True, (255, 255, 255), (0,0,0))
            ctx.screen.blit(text_surface, (40, 40))
            pygame.draw.circle(ctx.screen, ColorGreen, center=core.to_pix(target_point), radius=15)
            arm.render()

            ############################## YOUR CODE GOES HERE ####################################
            #######################################################################################
            # UA: Код для обчислення зворотної кінематики та малювання має бути тут               #
            # EN: Your code for IK and drawing goes here.                                         #
            #######################################################################################
            #######################################################################################
            # controller.move_to_angles([your predicted angles])

            # Calculate IK for current target position
            target_angles = inverse_kinematics(target_point, arm, max_iterations=100, tolerance=0.01)
            
            if controller.is_idle():
                controller.move_to_angles(target_angles)
            else:
                controller.step()

            #######################################################################################
            #######################################################################################

            # Make Box2D simulate the physics
            ctx.world.Step(TIME_STEP, 10, 10)
          
            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Terminating...")
    pygame.quit()