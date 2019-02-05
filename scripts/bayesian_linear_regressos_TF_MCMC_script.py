import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

sns.set()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
np.random.seed(111)
tf.set_random_seed(111)

from data_generator import DataGenerator
from HMC_parameters import HMC_parameters
from bayesian_linear_regressor_TF_MCMC import BayesianLinearRegressor_TF_MCMC


data = DataGenerator()
data.show()


print('True bias: %0.2f' % data.b_true)
print('True weights: ', data.w_true[:, 0])

hmc_parameter = HMC_parameters(data.number_of_dimensions)

bayesian_linear_regressor = BayesianLinearRegressor_TF_MCMC(hmc_parameter)
bayesian_linear_regressor.sample_posterior(data)


# Samples after burn-in
coeffs_samples, bias_samples, noise_std_samples, accepted_samples = bayesian_linear_regressor.return_samples()


print('Acceptance rate: %0.1f%%' % (100 * np.mean(accepted_samples)))


from chain_plotter import ChainPlotter

ChainPlotter.plot(data, coeffs_samples, bias_samples, noise_std_samples)

# It looks like our model accurately recovered the true parameters used to generate the data!

# ### Predictive Distribution
#
# To "criticize" our model, we can take a look at the posterior predictive distributions on held-out (validation) data.  The posterior predictive distribution is the distribution of $y$ values which our model predicts for a given held-out $x$, if we assume that the true parameter values follow the probability distribution that we computed using the non-held-out data (the posterior).  That is, it's how likely any given $y$ value is for a new $x$ value, if we incorporate all our sources of uncertainty.
#
# To look at the posterior predictive distributions, we need held-out data, so we'll first generate some validation data:

# In[16]:


# Generate held out data
N_val = 1000
x_val = np.random.randn(N_val, data.number_of_dimensions).astype(np.float32)
noise = data.noise_std_true * np.random.randn(N_val, 1).astype(np.float32)
y_val = np.matmul(x_val, data.w_true) + data.b_true + noise


# Then we can compute the predictive distributions.  We'll draw one sample from our probabilistic model per MCMC sample from the posterior distribution (though we could do more).
#
# TensorFlow Probability (and Edward) provide a method to do this they call "intercepting", which allows the user to set the value of the model parameters, and then draw a sample from the model.  Unfortunately this method isn't well-suited to drawing many samples each with different parameter values (i.e. it takes a long time), so we'll just do it manually.
#
# In the figure below, each plot corresponds to a different validation datapoint (I've only plotted 8 out of the 1000 we generated), the vertical lines show the true value of $y$ for that datapoint, and the distributions show the predictive distristribution (our guess for how likely each value of $y$ is given our model and our uncertainty as to the model's parameters).

# In[17]:


def ind_prediction_distribution(X):
    '''Compute the prediction distribution for an individual validation example'''
    predictions = np.matmul(X, coeffs_samples.transpose()) + bias_samples[:, 0]
    noise = noise_std_samples[:, 0] * np.random.randn(noise_std_samples.shape[0])
    return predictions + noise


# Compute prediction distribution for all validation samples
Nmcmc = coeffs_samples.shape[0]
prediction_distribution = np.zeros((N_val, Nmcmc))
for i in range(N_val):
    prediction_distribution[i, :] = ind_prediction_distribution(x_val[i, :])

# Plot random datapoints and their prediction intervals
fig, axes = plt.subplots(4, 2, sharex='all')
for i in range(4):
    for j in range(2):
        ix = i * 2 + j
        pred_dist = prediction_distribution[ix, :]
        sns.kdeplot(pred_dist, shade=True, ax=axes[i][j])
        axes[i][j].axvline(x=y_val[ix, 0])
    axes[i][0].set_ylabel('p(y)')

axes[3][0].set_xlabel('y')
axes[3][1].set_xlabel('y')

# We can also take the mean of each posterior predictive distribution, and compute the residuals (difference between the mean of each held-out datapoint's true $y$ value and the mean of that datapoint's posterior predictive distribution).

# In[18]:


# Plot the residual distribution
plt.figure()
y_pred = np.mean(prediction_distribution, axis=1)
residuals = y_val[:, 0] - y_pred
plt.hist(residuals, bins=30, density=True)
xx = np.linspace(-6, 6, 200)
plt.plot(xx, norm.pdf(xx, scale=np.std(residuals)))
plt.title('Residuals')
plt.xlabel('y_true - y_est')
plt.show()

# We used a normal distribution to model the error, so the residuals should be normally-distributed.  The residuals look pretty good normally-distributed, but if they hadn't, we might have wanted to change the type of distribution used to model noise.
#
# To asses how accurate our uncertainty estimates are, we can compute the coverage of the 95% interval.  That is, how often does the true $y$ value actually fall within the 95% interval of our posterior predictive distribution?  If our model is accurately capturing its uncertainty, then 95% of the true values should fall within the 95% interval of their posterior predictive distributions.

# In[19]:


# Compute proportion of estimates on validation data
# which fall within the 95% prediction interval
q0 = 2.5
q1 = 97.5
within_conf_int = np.zeros(N_val)
for i in range(N_val):
    pred_dist = prediction_distribution[i, :]
    p0 = np.percentile(pred_dist, q0)
    p1 = np.percentile(pred_dist, q1)
    if p0 <= y_val[i] and p1 > y_val[i]:
        within_conf_int[i] = 1

print('%0.1f %% of validation samples are w/i the %0.1f %% prediction interval'
      % (100 * np.mean(within_conf_int), q1 - q0))