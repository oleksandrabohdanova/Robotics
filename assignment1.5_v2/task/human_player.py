import pygame
from pygame.locals import *

from task.player import Player


class HumanPlayer(Player):
    def __init__(self):
        self.name = "Human"
        self.anim_id = 0
        self.alpha = 200
        super().__init__()

    def act(self, obs):
        thruster_left = self.thruster_mean
        thruster_right = self.thruster_mean
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_UP]:
            thruster_left += self.thruster_amplitude
            thruster_right += self.thruster_amplitude
        if pressed_keys[K_DOWN]:
            thruster_left -= self.thruster_amplitude
            thruster_right -= self.thruster_amplitude
        if pressed_keys[K_LEFT]:
            thruster_left -= self.diff_amplitude
        if pressed_keys[K_RIGHT]:
            thruster_right -= self.diff_amplitude
        return thruster_left, thruster_right
