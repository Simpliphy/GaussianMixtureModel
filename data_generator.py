import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataGenerator(object):

    def __init__(self):

        self._number_of_clusters = 3
        self._number_of_datapoints = 100
        #self._noise_std_true = 1.0

        self.prob_per_cluster = np.array([0.3, 0.3, 0.4])
        mean_per_cluster = np.array([1, 4, 8])
        # Generate data
        #self._b_true = np.random.randn(1).astype(np.float32)  # bias (alpha)
        #self._w_true = np.random.randn(self.number_of_dimensions, 1).astype(np.float32)  # weights (beta)
        p_assignment = np.random.uniform(0,1, size=self._number_of_datapoints)

        cluster_indexes = []
        upper_limit_probability = np.cumsum(self.prob_per_cluster)
        upper_limit_probability = np.concatenate((np.array([0.0]),upper_limit_probability))
        for index in range(self._number_of_clusters):
            lower_limit = upper_limit_probability[index]
            upper_limit = upper_limit_probability[index+1]
            indexes_true = np.argwhere((lower_limit <= p_assignment) & (p_assignment < upper_limit))

            cluster_indexes.append(np.ravel(indexes_true))

        self._x = np.zeros(self._number_of_datapoints)

        for index_cluster, list_indexes_for_cluster in enumerate(cluster_indexes):

            print(index_cluster)
            values = np.random.normal(loc=mean_per_cluster[index_cluster],
                             scale=1.0,
                             size=len(list_indexes_for_cluster))
            print(values.shape)
            print(self._x[list_indexes_for_cluster])
            self._x[list_indexes_for_cluster] = values

        #self._x = np.random.randn(self.number_of_datapoints, self.number_of_dimensions).astype(np.float32)
        #self._noise = self.noise_std_true * np.random.randn(self.number_of_datapoints, 1).astype(np.float32)
        #self._y = np.matmul(self.x, self.w_true) + self.b_true + self.noise

        #self.N_val = 1000
        #self.x_val = np.random.randn(self.N_val, self.number_of_dimensions).astype(np.float32)
        #self.noise_val = self.noise_std_true * np.random.randn(self.N_val, 1).astype(np.float32)
        #self.y_val = np.matmul(self.x_val, self.w_true) + self.b_true + self.noise_val


    @property
    def number_of_clusters(self):
        return self._number_of_clusters

    @property
    def number_of_datapoints(self):
        return self._number_of_datapoints



    @property
    def x(self):
        return self._x



    def show(self):

        plt.hist(self.x)
        plt.show()




