import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        self.model1 = ManiuplatorModel(Tp, 0.1, 0.05)
        self.model2 = ManiuplatorModel(Tp, 0.01, 0.01)
        self.model3 = ManiuplatorModel(Tp, 1.0, 0.3)

        self.models = [self.model1, self.model2, self.model3]
        self.i = 0
        self.u = np.array([[0], [0]])
        self.Kp = np.array([[15, 0], [0, 15]])
        self.Kd = np.array([[20, 0], [0, 40]])

    def choose_model(self, x):
        lastError = np.inf
        for index, model in enumerate(self.models):
            y = model.M(x) @ self.u + model.C(x) @ x[2:]
            error = np.sum(abs(x[:2] - y[:2]))
            if error < lastError:
                lastError = error
                self.i = index

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        v = q_r_ddot + self.Kd @ (q_r - q_dot) - self.Kp @ (q - q_r)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
