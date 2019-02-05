from GaussianSoftClustering import GaussianSoftClustering
from data_generator import DataGenerator
from GaussianMixtureFunctor_1d import GaussianMixtureFunctor_1d
import numpy as np
import matplotlib.pyplot as plt

data = DataGenerator()
data.show()

gmm_model = GaussianSoftClustering()

observations = data.x.copy()
observations = observations.reshape((observations.shape[0], 1))

best_loss, pi, mu, sigma, gamma = gmm_model.train_EM(observations,
                                                     number_of_clusters=3,
                                                     restarts=200,
                                                     max_iter=100)

mu = mu.ravel()
sigma = sigma.ravel()

mixture = GaussianMixtureFunctor_1d(pi, mu, sigma)

x = np.linspace(-1, 11, 100)
density = np.array([mixture(value) for value in x])

plt.grid(True)
plt.plot(x, density,  linewidth=3, label="density calculated")
plt.hist(data.x, normed=True, color="blue", bins=20, alpha = 0.5, label="data simulated")
plt.ylabel("density")
plt.xlabel("feature value (x)")
plt.title("Gaussian mixture obtained using Expectation-Maximization")
plt.legend()
plt.show()


