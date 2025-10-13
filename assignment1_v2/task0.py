import time
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import robo_algo.core as core
from robo_algo.core import Color, ColorGreen
from robo_algo.arm import RoboticArm, RoboticArmPlotter
from robo_algo.arm_controller import ArmController
from robo_algo.constants import *

import numpy as np


TEST_RUN = False # Set to True to test your code

def forward_kinematics(arm: RoboticArm, angles=None):
    ############################################################
    # UA: Реалізуйте пряму кінематику тут.
    # EN: Implement forward kinematics logic here.
    ############################################################

    base_position = np.array(arm.joints[0].position)
    link_lengths = np.array(arm.link_lengths)
    if angles is None:
        angles = np.array(arm.get_angles())
    
    x = base_position[0]
    y = base_position[1]
    
    cumulative_angle = 0
    
    for i in range(len(link_lengths)):
        cumulative_angle += angles[i]
        
        x += link_lengths[i] * np.cos(cumulative_angle)
        y += link_lengths[i] * np.sin(cumulative_angle)
    
    end_effector_position = np.array([x, y])
    
    ############################################################
    ############################################################

    if end_effector_position is None:
        end_effector_position = np.array([3., 3.])
        
    return end_effector_position[:2]


if __name__ == "__main__":
    ctx = core.RenderingContext("Task 0 - visualization")
    arm_dummy = RoboticArmPlotter(
        ctx,
        joint0_position=np.array([30, 30]),
        link_lengths=[2],
        link_angles=[np.deg2rad(0)],
        thickness=0.1,
        color=Color(127, 127, 127, 255),
        joint_radius=0.3,
        joint_color=Color(200, 200, 200, 255),
    )

    arm1 = RoboticArmPlotter(
        ctx,
        joint0_position=np.array([8, 4]),
        link_lengths=[1, 1, 2, 1, 2],
        link_angles=[np.deg2rad(0), np.deg2rad(10), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)],
        thickness=0.1,
        color=Color(127, 127, 127, 255),
        joint_radius=0.3,
        joint_color=Color(200, 200, 200, 255),
    )
    arm_dummy.start_drawing(ColorGreen)
    arm1.start_drawing()
    arm1_controller = ArmController(arm1)
    arm1_controller.move_to_angles(
        [np.deg2rad(170), np.deg2rad(50), np.deg2rad(100), np.deg2rad(170), np.deg2rad(50)],
    )

    running = True
    spf_running_mean = 0
    coef = 0
    a = 0
    try:
        while running:
            
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                   
                    running = False

            ms_start = time.perf_counter_ns() / 1000
            ctx.screen.fill((0, 0, 0, 0))

            if a < 10:
                arm_dummy.render()
                arm1.render()
                arm1_controller.step()
                arm1.draw()
                if arm1_controller.is_idle():
                    a += 1
                    if a % 2 == 0:
                        arm1_controller.move_to_angles(
                            [np.deg2rad(190), np.deg2rad(60), np.deg2rad(110), np.deg2rad(190), np.deg2rad(60)]
                        )
                    else:
                        arm1_controller.move_to_angles(
                            [np.deg2rad(180), np.deg2rad(40 + a*25), np.deg2rad(100), np.deg2rad(a*10), np.deg2rad(a*10)]
                        )
                
                pos = forward_kinematics(arm1)
                pos_gt = arm1_controller.get_end_effector_position()
                arm_dummy.draw(np.array(pos)+0.01)
                if TEST_RUN:
                    assert np.isclose(pos, pos_gt, atol=0.1).all()
            else:
                running = False


            ctx.world.Step(TIME_STEP, 10, 10)

            pygame.display.flip()
            ctx.clock.tick(TARGET_FPS)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Terminating...")
    pygame.quit()