import numpy as np

from task.player import Player


class PID:
    def __init__(
        self,
        P: float,
        I: float,
        D: float,
        saturation_max=np.inf,
        saturation_min=-np.inf,
    ):
        self.P = P
        self.I = I
        self.D = D
        self.saturation_max = saturation_max
        self.saturation_min = saturation_min

        ###############################################
        # You can initialize accumulators or temporary
        # variables here.
        ###############################################
        # YOUR CODE HERE
        ###############################################

    def compute(self, error: float, dt: float):
        ###############################################
        # Compute PID controller output
        ###############################################
        # YOUR CODE HERE
        output = None
        ###############################################

        output = np.clip(output, self.saturation_min, self.saturation_max)
        return output


class PIDPlayer(Player):
    def __init__(self):
        self.name = "PID"
        self.alpha = 200
        self.anim_id = 1
        super().__init__()

        self.thruster_amplitude *= 3
        self.diff_amplitude *= 3

        ###############################################
        # You can tune PID parameters here.
        # I provided some good values, but feel
        # free to tune them yourself for better
        # performance!
        ###############################################
        # cascade 1
        self.x_pid = PID(0.2 / 3, 0, 0.2, 25, -25)
        self.angle_pid = PID(0.02 / 3, 0, 0.01 / 3, 1, -1)
        # cascade 2
        self.y_pid = PID(2.5 / 3, 0, 1.5 / 3, 100, -100)
        self.y_speed_pid = PID(1 / 3, 0, 0, 1, -1)
        ###############################################

    def act(self, obs):
        thruster_left = self.thruster_mean
        thruster_right = self.thruster_mean

        # Input data: x_target - balloon X coordinate
        #             y_target - balloon Y coordinate
        #             x_drone - drone X coordinate
        #             y_drone - drone Y coordinate
        #             y_drone_speed - drone Y speed
        #             drone_angle - drone's pitch angle (\theta)
        (
            x_target,
            y_target,
            x_drone,
            y_drone,
            _,
            y_drone_speed,
            drone_angle,
            _,
        ) = obs

        ###############################################
        # Implement PID Cascade 1
        ###############################################
        # YOUR CODE HERE
        error_y = None
        y_drone_speed_setpoint = None
        error_y_drone_speed = None
        thrust_0 = self.y_speed_pid.compute(-error_y_drone_speed, self.dt)
        ###############################################

        ###############################################
        # Implement PID Cascade 2
        ###############################################
        # YOUR CODE HERE
        error_x = None
        angle_setpoint = None
        error_angle = None
        thrust_1 = self.angle_pid.compute(-error_angle, self.dt)
        ###############################################

        # calculating motor thrusts
        thruster_left += thrust_0 * self.thruster_amplitude
        thruster_right += thrust_0 * self.thruster_amplitude

        thruster_left += thrust_1 * self.diff_amplitude
        thruster_right -= thrust_1 * self.diff_amplitude

        return thruster_left, thruster_right
