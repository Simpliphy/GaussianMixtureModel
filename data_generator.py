import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataGenerator(object):

    def __init__(self):

        self._number_of_dimensions = 4
        self._number_of_datapoints = 100
        self._noise_std_true = 1.0

        # Generate data
        self._b_true = np.random.randn(1).astype(np.float32)  # bias (alpha)
        self._w_true = np.random.randn(self.number_of_dimensions, 1).astype(np.float32)  # weights (beta)
        self._x = np.random.randn(self.number_of_datapoints, self.number_of_dimensions).astype(np.float32)
        self._noise = self.noise_std_true * np.random.randn(self.number_of_datapoints, 1).astype(np.float32)
        self._y = np.matmul(self.x, self.w_true) + self.b_true + self.noise

        self.N_val = 1000
        self.x_val = np.random.randn(self.N_val, self.number_of_dimensions).astype(np.float32)
        self.noise_val = self.noise_std_true * np.random.randn(self.N_val, 1).astype(np.float32)
        self.y_val = np.matmul(self.x_val, self.w_true) + self.b_true + self.noise_val

    @property
    def b_true(self):
        return self._b_true

    @property
    def number_of_dimensions(self):
        return self._number_of_dimensions

    @property
    def number_of_datapoints(self):
        return self._number_of_datapoints

    @property
    def noise_std_true(self):
        return self._noise_std_true

    @property
    def x(self):
        return self._x

    @property
    def w_true(self):
        return self._w_true

    @property
    def noise(self):
        return self._noise

    @property
    def y(self):
        return self._y

    def show(self):

        fig, axes = plt.subplots(int(np.ceil(self.number_of_dimensions / 2)), 2, sharex=True)
        fig.set_size_inches(6.4, 6)
        for i in range(self.number_of_dimensions):
            t_ax = axes[int(i / 2), i % 2]  # this axis
            sns.regplot(self.x[:, i], self.y[:, 0], ax=t_ax)
            t_ax.set_ylabel('y')
            t_ax.set_xlabel('x[%d]' % (i + 1))
        plt.show()




