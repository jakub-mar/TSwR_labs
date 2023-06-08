import numpy as np

from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp, 0.1, 0.05)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array(
            [
                [3 * p[0], 0],
                [0, 3 * p[1]],
                [3 * p[0] ** 2, 0],
                [0, 3 * p[1] ** 2],
                [p[0] ** 3, 0],
                [0, p[1] ** 3],
            ]
        )
        self.W = np.concatenate((np.eye(2), np.zeros((2, 4))), axis=1)
        self.A = np.zeros((6, 6))
        self.A[0, 2] = 1
        self.A[1, 3] = 1
        self.A[2, 4] = 1
        self.A[3, 5] = 1
        self.B = np.zeros((6, 2))
        self.eso = ESO(self.A, self.B, self.W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        x = np.array([q, q_dot])
        M = self.model.M(x)
        M_inv = np.linalg.inv(M)
        C = self.model.C(x)

        self.A[2:4, 2:4] = -(M_inv @ C)
        self.B[2:4, 0:2] = M_inv
        self.eso.A = self.A
        self.eso.B = self.B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        M = self.model.M(x)
        C = self.model.C(x)

        x_h, x_hd, f = (
            self.eso.get_state()[:2],
            self.eso.get_state()[2:4],
            self.eso.get_state()[4:],
        )

        v = q_d_ddot + self.Kd @ (q_d_dot - x_hd) + self.Kp @ (q_d - q)
        u = M @ (v - f) + C @ x_hd
        self.update_params(x_h, x_hd)
        self.eso.update(q.reshape(len(q), 1), u.reshape(len(u), 1))
        return u
