from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []
        # self.x_h = np.zeros_like(self.state[0 : int(len(self.state) / 3)])

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        # z_hd = (
        #     np.dot(self.A, self.state)
        #     + np.squeeze(np.dot(self.B, u).T)
        #     + np.dot(
        #         self.L, np.squeeze(q - np.array([np.dot(self.W, self.state.T)]).T).T
        #     )
        # )
        z_hd = (
            self.A @ np.reshape(self.state, (len(self.state), 1))
            + self.B @ np.atleast_2d(u)
            + (self.L @ (q - self.W @ np.reshape(self.state, (len(self.state), 1))))
        )

        self.state = (self.state + self.Tp * np.reshape(z_hd, (1, len(z_hd))))[0]

        # `self.x_h` = zh[:2] if z_hd > 3 else zh[0]

    def get_state(self):
        return self.state
