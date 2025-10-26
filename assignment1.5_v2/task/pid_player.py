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
        self.accumulated_error = 0.0
        self.previous_error_value = 0.0
        self.derivative_history = [0.0, 0.0, 0.0]
        ###############################################

    def reset_accumulator(self):
        """Reset integral accumulator to prevent windup"""
        self.accumulated_error = 0.0

    def compute(self, error: float, dt: float):
        ###############################################
        # Compute PID controller output
        ###############################################
        if dt > 0.001:
            current_derivative = (error - self.previous_error_value) / dt
            self.derivative_history.pop(0)
            self.derivative_history.append(current_derivative)
            avg_derivative = sum(self.derivative_history) / len(self.derivative_history)
        else:
            avg_derivative = 0.0
        
        self.accumulated_error += error * dt
        max_integral = 50.0
        self.accumulated_error = np.clip(self.accumulated_error, -max_integral, max_integral)
        
        P_component = self.P * error
        I_component = self.I * self.accumulated_error
        D_component = self.D * avg_derivative
        
        self.previous_error_value = error
        
        output = P_component + I_component + D_component
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
        self.x_pid = PID(0.085, 0, 0.25, 25, -25)
        self.angle_pid = PID(0.0075, 0, 0.0045, 1, -1)
        # cascade 2
        self.y_pid = PID(1.15, 0, 0.7, 100, -100)
        self.y_speed_pid = PID(0.42, 0, 0, 1, -1)
        
        self.previous_target = None
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

        if self.previous_target is not None:
            prev_x, prev_y = self.previous_target
            if abs(x_target - prev_x) > 50 or abs(y_target - prev_y) > 50:
                self.x_pid.reset_accumulator()
                self.y_pid.reset_accumulator()
        self.previous_target = (x_target, y_target)

        ###############################################
        # Implement PID Cascade 1
        ###############################################
        error_y = y_target - y_drone
        y_drone_speed_setpoint = self.y_pid.compute(error_y, self.dt)
        error_y_drone_speed = y_drone_speed_setpoint - y_drone_speed
        thrust_0 = self.y_speed_pid.compute(-error_y_drone_speed, self.dt)
        ###############################################

        ###############################################
        # Implement PID Cascade 2
        ###############################################
        error_x = x_drone - x_target
        angle_setpoint = self.x_pid.compute(error_x, self.dt)
        error_angle = angle_setpoint - drone_angle
        thrust_1 = self.angle_pid.compute(-error_angle, self.dt)
        ###############################################

        # calculating motor thrusts
        thruster_left += thrust_0 * self.thruster_amplitude
        thruster_right += thrust_0 * self.thruster_amplitude

        thruster_left += thrust_1 * self.diff_amplitude
        thruster_right -= thrust_1 * self.diff_amplitude

        return thruster_left, thruster_right