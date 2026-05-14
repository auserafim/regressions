from re import M

import numpy as np


class MultipleLinearRegression:
    def __init__(self):
        b = None

    def b(self, y: np.matrix, x: np.matrix):
        x_transposed = np.transpose(x)
        return x_transposed * y(np.linalg.inv(x_transposed * x))


if __name__ == "__main__":
    mlr = MultipleLinearRegression()
    pass
