import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], np.float32)
        self.B = np.array([[0], [self.b], [0]], np.float32)
        self.L = np.array([[3 * p], [3 * p**2], [p**3]], np.float32)
        W = np.array([[1, 0, 0]], np.float32)

        self.eso = ESO(A, self.B, W, self.L, q0, Tp)
        self.model = ManiuplatorModel(Tp, 0.1, 0.05)

    def set_b(self, b):
        self.b = b
        self.B = np.array([[0], [b], [0]], np.float32)
        self.eso.set_B(self.B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, i):
        q = x[0]
        x_h, x_hd, f = self.eso.get_state()

        v = q_d_ddot + self.kd * (q_d_dot - x_hd) + self.kp * (q_d - q)
        u = (v - f) / self.b

        self.eso.update(q, u)
        M = self.model.M(np.concatenate([[x_h], [0.0, 0.0, 0.0]], axis=0))
        self.set_b(np.linalg.inv(M)[i, i])

        return u
