import pygame
from robo_algo.constants import *

from typing import NamedTuple
import numpy as np
import Box2D
from Box2D.b2 import world, polygonShape


class Color(NamedTuple):
    r: int
    g: int
    b: int
    a: int

ColorGreen = Color(0, 200, 0, 255)
ColorRed = Color(200, 0, 0, 255)
ColorBlue = Color(0, 0, 200, 255)
ColorYellow = Color(200, 200, 0, 255)


class RenderingContext:

    def __init__(self, name: str):
        # --- pygame setup ---
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption(name)
        self.clock = pygame.time.Clock()

        # --- pybox2d world setup ---
        # Create the world
        self.world = world(gravity=(0, -10), doSleep=True)

        # And a static body to hold the ground shape
        self.ground_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=polygonShape(box=(50, 1)),
        )

    def draw_polygon(self, body, fixture, color: Color):
        shape = fixture.shape
        vertices = [to_pix(body.transform * v) for v in shape.vertices]
        pygame.draw.polygon(self.screen, color, points=vertices)

    def draw_circle(self, body, fixture, color: Color):
        shape = fixture.shape
        position = to_pix(body.transform * shape.pos)
        pygame.draw.circle(self.screen, color, position, int(shape.radius * PPM))

    def draw_end_effector(self, body, fixture, color: Color):
        shape = fixture.shape
        vertices = [to_pix(body.transform * v) for v in shape.vertices]
        pygame.draw.circle(self.screen, color, vertices)
    
    def draw_lines(self, points_pix, color: Color, width=1):
        pygame.draw.lines(self.screen, color, closed=False, points=points_pix, width=width)


_a = np.array([1, -1]) * PPM
_b = np.array([0, SCREEN_HEIGHT])
def to_pix(arr):
    global _a, _b
    return list(map(int, (_a*arr + _b)))

def from_pix(val):
    global _a, _b
    return (val - _b) / _a

def np_to_pix(np_arr):
    global _a, _b
    return (_a*np_arr + _b).astype(np.int32)
