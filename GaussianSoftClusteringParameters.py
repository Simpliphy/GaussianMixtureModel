import numpy as np


class GaussianSoftClusteringParameters(object):

    def __init__(self):

        self.gamma = None
        self.pi = None
        self.mu = None
        self.sigma = None

    def initialize_parameters(self, number_of_clusters, number_of_features, number_of_observations):

        pi = 1 / float(number_of_clusters) * np.ones(number_of_clusters)
        mu = np.random.randn(number_of_clusters, number_of_features)
        sigma = np.zeros((number_of_clusters, number_of_features, number_of_features))
        sigma[...] = np.identity(number_of_features)

        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.gamma = np.zeros((number_of_observations, number_of_clusters))
