

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pymc3 as pm
import theano.tensor as tt
from data_generator import DataGenerator
from GaussianMixtureFunctor_1d import GaussianMixtureFunctor_1d

sns.set_context('paper')
sns.set_style('darkgrid')

data_generated = DataGenerator()
data_generated.show()

observations = data_generated.x.copy()

ndata = len(observations)
data = observations
k = 3

# setup model
model = pm.Model()
with model:

    # cluster sizes
    p = pm.Dirichlet('p', a=np.array([1., 1., 1.]), shape=k)

    # ensure all clusters have some points
    p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(p) < .1, -np.inf, 0))


    # cluster centers
    means = pm.Normal('means', mu=[0, 0, 0], sd=2.0, shape=k)

    # break symmetry
    order_means_potential = pm.Potential('order_means_potential',
                                         tt.switch(means[1]-means[0] < 0, -np.inf, 0)
                                         + tt.switch(means[2]-means[1] < 0, -np.inf, 0))

    # measurement error
    sd = pm.HalfCauchy('sd',  beta=2, shape=k)

    # latent cluster of each observation
    category = pm.Categorical('category',
                              p=p,
                              shape=ndata)

    # likelihood for each observed value
    points = pm.Normal('obs',
                       mu=means[category],
                       sd=sd[category],
                       observed=data)

number_of_samples = 50000
with model:
    #step1 = pm.NUTS(vars=[p, sd, means])
    #step2 = pm.CategoricalGibbsMetropolis(vars=[category], proposal='proportional')
    tr = pm.sample(number_of_samples)#, step=[step1, step2])

end_burning_index=10000
pm.plots.traceplot(tr[end_burning_index::5], ['p', 'sd', 'means'])
plt.show()

pm.autocorrplot(tr[end_burning_index::5], varnames=['sd'])
plt.show()

i=0
plt.plot(tr['category'][end_burning_index::5, i], drawstyle='steps-mid')
plt.axis(ymin=-.1, ymax=2.1)
plt.show()


mu = np.mean(tr.get_values('means', burn=end_burning_index, combine=True), axis=0)
sigma = np.mean(tr.get_values('sd', burn=end_burning_index, combine=True), axis=0)
categorical_p = np.mean(tr.get_values('p', burn=end_burning_index, combine=True), axis=0)


mixture = GaussianMixtureFunctor_1d(categorical_p, mu, sigma)

x = np.linspace(-1, 11, 100)
density = np.array([mixture(value) for value in x])

plt.grid(True)
plt.plot(x, density,  linewidth=3, label="density calculated")
plt.hist(data_generated.x, normed=True, color="blue", bins=20, alpha=0.5, label="data simulated")
plt.ylabel("density")
plt.xlabel("feature value (x)")
plt.title("Gaussian mixture obtained using Expectation-Maximization")
plt.legend()
plt.show()


