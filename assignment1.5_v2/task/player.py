"""
2D Quadcopter implementation based on Alexandre Sajus's work.
"""


class Player:
    def __init__(self):
        self.thruster_mean = 0.04 * 3600
        self.thruster_amplitude = 0.04 * 3600
        self.diff_amplitude = 0.003 * 3600
        self.angle, self.angular_speed, self.angular_acceleration = (0, 0, 0)
        self.x_position, self.x_speed, self.x_acceleration = (400, 0, 0)
        self.y_position, self.y_speed, self.y_acceleration = (400, 0, 0)
        self.target_counter = 0
        self.dead = False
        self.respawn_timer = 3
        self.dt = 1 / 60
