import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp, m3, r3):
        self.model = ManiuplatorModel(Tp, m3, r3)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x

        q = x[:2]
        q_dot = x[2:4]
        Kd = np.array([0.1, 0.1])
        Kp = np.array([0.1, 0.1])
        v = q_r_ddot + Kd @ (q_dot - q_r_dot) + Kp @ (q - q_r)
        tau = self.model.M(x) @ v + self.model.C(x) @ q_dot
        return tau
