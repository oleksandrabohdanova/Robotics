from typing import List
import numpy as np
import numpy.typing as npt

from robo_algo.constants import *
import robo_algo.core as core
from robo_algo.core import Color, ColorRed
from Box2D.b2 import polygonShape, circleShape


def _normalize_angle(angle):
    return np.mod(angle, 2*np.pi)

class RoboticArm:
    """
    Data structure for storing positions and characteristics of an arm and drawing the arm on the screen.
    """

    def __init__(
        self,
        ctx,
        joint0_position: List[float],
        link_lengths: List[float],
        link_angles: List[float],
        thickness: float,
        color: Color,
        joint_radius: float,
        joint_color: Color,
        end_effector_color: Color = ColorRed,
    ) -> None:

        self.ctx = ctx

        self.p0 = np.array(joint0_position).astype(float)
        self._link_angles = np.array(link_angles).astype(float)
        self.link_lengths = np.array(link_lengths).astype(float)

        assert len(link_angles) == len(link_lengths)
        self.num_links = len(link_lengths)

        self.links  = [None] * self.num_links
        self.joints = [None] * (self.num_links + 1)

        self.color = color
        self.joint_color = joint_color
        self.end_effector_color = end_effector_color
        self.joint_radius = joint_radius
        self.thickness = thickness

        self._recalc_links_and_joints()
    
    def set_angles(self, link_angles: List[float]):
        """
        Use this function if you need to move robot's
        arm immediately. It is not a motion, just a
        change of position.
        """
        assert len(self._link_angles) == len(link_angles)
        self._link_angles = np.array(link_angles).astype(float)
        self._link_angles = _normalize_angle(self._link_angles)
        self._recalc_links_and_joints()
    
    def set_angle(self, i_angle, value):
        self._link_angles[i_angle] = _normalize_angle(float(value))
        self._recalc_links_and_joints()

    def get_angles(self) -> npt.NDArray[np.float32]:
        """
        If you want to get current joint angles,
        this is the function you are looking for!
        Do not access _link_angles directly!
        """
        return np.array(self._link_angles).copy()
    
    def _recalc_links_and_joints(self):
        self.link_cumm_angles = np.cumsum(self._link_angles)
        self.link_end_points = np.stack(
            (self.link_lengths * np.cos(self.link_cumm_angles),
            self.link_lengths * np.sin(self.link_cumm_angles)), axis=1)
        self.link_end_points = np.atleast_2d(self.link_end_points)
        self.link_end_points[0] += self.p0
        self.link_end_points = np.cumsum(self.link_end_points, axis=0)
        self.link_begin_points = np.empty_like(self.link_end_points)
        self.link_begin_points[1:] = self.link_end_points[:-1]
        self.link_begin_points[0] = self.p0
        self.link_center_points = (self.link_end_points + self.link_begin_points) * 0.5
        if any(map(lambda link: link is None, self.links)):
            self.joints[0] = self.ctx.world.CreateStaticBody(
                position=list(self.p0),
                shapes=circleShape(radius=self.joint_radius),
            )

            for i_link in range(self.num_links):
                p_center = self.link_center_points[i_link]
                p_end = self.link_end_points[i_link]
                angle = self.link_cumm_angles[i_link]
                length = self.link_lengths[i_link]

                self.links[i_link] = self.ctx.world.CreateStaticBody(
                    position=list(p_center),
                    angle=angle,
                    shapes=polygonShape(box=(length/2, self.thickness)),
                )
                self.joints[i_link + 1] = self.ctx.world.CreateStaticBody(
                    position=list(p_end),
                    shapes=circleShape(radius=self.joint_radius),
                )
            self.joints[-1] = self.ctx.world.CreateStaticBody(
                position=list(p_end),
                shapes=circleShape(radius=self.joint_radius / 2)
            )
        else:
            for i_link in range(self.num_links):
                p_center = self.link_center_points[i_link]
                p_end = self.link_end_points[i_link]
                angle = self.link_cumm_angles[i_link]

                self.links[i_link].angle = angle
                self.links[i_link].position = p_center
                self.joints[i_link + 1].position = p_end
    
    def render(self):
        """
        Draw the arm on the screen.
        """
        for i_link in range(self.num_links):
            self.ctx.draw_polygon(self.links[i_link], self.links[i_link].fixtures[0], self.color)
        for i_link in range(self.num_links ):
            self.ctx.draw_circle(self.joints[i_link], self.joints[i_link].fixtures[0], self.joint_color)
        # NOTE(aty): joints[-1] is an end-effector
        self.ctx.draw_circle(self.joints[-1], self.joints[-1].fixtures[0], self.end_effector_color)



class RoboticArmPlotter(RoboticArm):
    def __init__(self, *args, **kwargs):
        """
        An arm + plotter capabilities in one object.
        See RoboticArm docs for arm functions.
        Example plotter usage:
        ```
        arm.start_drawing()
        while running:
            arm.draw()
        arm.stop_drawing()
        """
        self.is_drawing = False
        self.points = []
        self.num_points = 0
        self.pix_threshold = 10
        self.max_num_points = 1000
        self.draw_color = ColorRed

        super().__init__(*args, **kwargs)

    def start_drawing(self, color=ColorRed):
        """
        Enable drawing mode, in which the arm's
        end effector leaves a colored trace (red by default).
        """
        self.points.append([])
        self.is_drawing = True
        self.draw_color = color

    def draw(self, point=None):
        """
        Use this function to actually draw a current point on
        the screen.
        """
        if point is None:
            point = self.joints[-1].position
        assert self.num_points < self.max_num_points, "Too many points. Something went wrong..."
        if self.is_drawing:
            point = core.np_to_pix(np.array(point))
            if len(self.points[-1]) > 0:
                if np.sum((self.points[-1][-1] - point)**2) > self.pix_threshold**2:
                    self.points[-1].append(point)
                    self.num_points += 1
            else:
                self.points[-1].append(point)
                self.num_points += 1

    def stop_drawing(self):
        """
        Disable drawing mode, in which the arm's
        end effector leaves a colored trace (red by default).
        """
        self.is_drawing = False
    
    def render(self):
        super().render()

        for line in self.points:
            if len(line) > 1:
                self.ctx.draw_lines(line, self.draw_color, width=2)


if __name__ == "__main__":
    import time
    import pygame
    from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

    ctx = core.RenderingContext("arm test")
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
    arm1 = RoboticArm(
        ctx,
        joint0_position=np.array([8, 8]),
        link_lengths=[2, 3, 4],
        link_angles=[np.deg2rad(0), np.deg2rad(10), np.deg2rad(0)],
        thickness=0.1,
        color=Color(127, 127, 127, 255),
        joint_radius=0.3,
        joint_color=Color(200, 200, 200, 255),
    )

    running = True
    spf_running_mean = 0
    coef = 0
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
            a0 = arm0.get_angles()[0]
            arm0.set_angle(0, a0 + np.pi/60)
            arm1.render()
            cur_angles = arm1.get_angles()
            arm1.set_angles([cur_angles[0] + np.pi/60,
                            cur_angles[1] + np.pi/30,
                            cur_angles[2] + np.pi/30])
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