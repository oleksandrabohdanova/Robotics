import time
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import robo_algo.core as core
from robo_algo.core import Color
from robo_algo.arm import RoboticArm, RoboticArmPlotter
from robo_algo.constants import *

import enum
import numpy as np


class ControllerState(enum.Enum):
    IDLE=0,  # no motion
    MOVING=1,  # in motion


class ArmController:
    def __init__(self, arm, max_velocity=np.deg2rad(0.5)):
        """
        This class defines a controller. A piece of logic used for a single purpose - interpolate
        between current joint angles and target joint angles passed to `move_to_angles`.
        The usual way is:
        ```
        ...
        while running:
            ...
            # inside while loop
            if arm_controller.is_idle():
                angles = compute_next_angles_with_ik_or_something_like_that(...)
                arm_controller.move_to_angle(angles)
            else:
                arm_controller.step()
        ```
        """
        self.arm = arm
        self.max_angular_velocity = max_velocity
        self._gen = None
        self.state = ControllerState.IDLE
    
    def get_end_effector_position(self):
        """!!!
        Get ground truth end effector position. Do not use it! Use your forward kinematics code instead!
        """
        return self.arm.joints[-1].position
    
    def move_to_angles(self, angles):
        """
        Start moving the arm by setting a set of target joint angle values (`angles`)
        After it is called, the `state` is set to `MOVING`. And until the `state` is not
        `IDLE` again (the motion stopped), a user must call `.step()` function and not
        do replanning.
        In other words, do not call `.move_to_angles(...)` when the state is not `IDLE`.
        You can check if state is idle by calling `.is_idle()`
        """
        a = angles
        def _gen():
            angles = np.array(a)
            cur_angles = self.arm.get_angles()
            cur_angles = np.array(cur_angles)

            num_steps = np.max(np.abs(angles - cur_angles)) / self.max_angular_velocity

            num_steps = max(num_steps, 2)
            for alpha in np.linspace(0.0, 1.0, num=int(num_steps), endpoint=True):
                next_angles = cur_angles*(1-alpha) + angles*alpha
                yield next_angles

        self._gen = _gen()
        self._next_angles = next(self._gen)
        self.state = ControllerState.MOVING
    
    def step(self):
        """
        Do a single step of a planned movement. Must be called **repeatedly** until the state is not
        `IDLE`
        """
        if self._gen is not None:
            try:
                self.arm.set_angles(self._next_angles)
                self._next_angles = next(self._gen)
            except StopIteration:
                self._gen = None
                self.state = ControllerState.IDLE
    
    def is_idle(self):
        """
        Returns True if there's no motion planned on the arm.
        """
        return self.state == ControllerState.IDLE

    def set_idle(self):
        """
        Stop motion and set state to IDLE. May be useful for jumping from one drawing shape to another.
        """
        self.state = ControllerState.IDLE


if __name__ == "__main__":
    ctx = core.RenderingContext("arm controller test")
    arm0 = RoboticArm(
        ctx,
        joint0_position=np.array([3, 8]),
        link_lengths=[2],
        link_angles=[np.deg2rad(0)],
        thickness=0.1,
        color=Color(127, 127, 127, 255),
        joint_radius=0.3,
        joint_color=Color(200, 200, 200, 255),
    )
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
    arm1_controller = ArmController(arm1)
    arm1_controller.move_to_angles(
        [np.deg2rad(170), np.deg2rad(50), np.deg2rad(100)],
    )

    running = True
    spf_running_mean = 0
    coef = 0
    a = 0
    try:
        while running:
            # Check the event queue
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    # The user closed the window or pressed escape
                    running = False

            ms_start = time.perf_counter_ns() / 1000
            ctx.screen.fill((0, 0, 0, 0))

            arm0.render()
            arm0.set_angle(0, arm0.get_angles()[0] + np.pi/60)
            arm1.render()
            arm1_controller.step()
            arm1.draw(arm1_controller.get_end_effector_position())
            if arm1_controller.is_idle():
                a += 1
                if a % 2 == 0:
                    arm1_controller.move_to_angles(
                        [np.deg2rad(190), np.deg2rad(60), np.deg2rad(110)],
                    )
                else:
                    arm1_controller.move_to_angles(
                        [np.deg2rad(180), np.deg2rad(40), np.deg2rad(100)],
                    )


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