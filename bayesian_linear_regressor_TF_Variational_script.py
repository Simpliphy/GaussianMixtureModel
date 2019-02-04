
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from Timer import Timer

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# The true coverage of the 95% interval is close to 95%, which means our model is pretty well-calibrated!


sns.set()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
np.random.seed(111)
tf.set_random_seed(111)

from data_generator import DataGenerator
from variational_parameters_bayesian_regressor import VariationalParameters_LinearRegressor
from bayesian_linear_regressior_TF_Var import VariationalLinearBayesianRegressor

data = DataGenerator()
data.show()
params = VariationalParameters_LinearRegressor()
varational_regressor = VariationalLinearBayesianRegressor(data, params)




w_post, b_post, n_post, prediction_dist_var = varational_regressor.fit()

# Plot value of weights over training
plt.figure()
for iW in range(data.number_of_dimensions):
  plt.plot(varational_regressor.weight_means[:,iW], label='Fit W[{}]'.format(iW),
           color=colors[iW])
  plt.hlines(data.w_true[iW], 0, varational_regressor.variational_parameters.max_steps, label='True W[{}]'.format(iW),
             color=colors[iW], linestyle='--')
plt.xlabel('Training Step')
plt.ylabel('Weight posterior mean')
plt.title('Weight parameters over training')
plt.legend()
plt.show()


# Similarly, we can visualize the standard deviation of the variational distributions over training:

# In[28]:


# Plot value of weight std devs over training
plt.figure()
for iW in range(data.number_of_dimensions):
  plt.plot(varational_regressor.weight_stds[:,iW], label='W[{}]'.format(iW),
           color=colors[iW])
plt.xlabel('Training Step')
plt.ylabel('Weight posterior std dev')
plt.title('Weight posterior std devs over training')
plt.legend()
plt.show()


# We'll also take a look at the bias and noise parameters over training, as well as the mean squared error and ELBO loss.

# In[29]:


# Plot value of bias over training
plt.figure()
plt.plot(varational_regressor.bias_means, label='fit')
plt.axhline(data.b_true, label='true', linestyle='--')
plt.xlabel('Training Step')
plt.ylabel('Bias posterior mean')
plt.title('Bias parameter over training')
plt.legend()
plt.show()


# In[30]:


# Plot value of bias std dev over training
plt.figure()
plt.plot(varational_regressor.bias_stds, label='fit')
plt.xlabel('Training Step')
plt.ylabel('Bias posterior std dev')
plt.title('Bias std dev over training')
plt.legend()
plt.show()


# In[31]:


# Plot value of noise std dev over training
plt.figure()
#plt.plot(noise_stds, label='fit')
plt.plot(varational_regressor.noise_means, label='fit')
plt.axhline(data.noise_std_true, label='true', linestyle='--')
plt.xlabel('Training Step')
plt.ylabel('Noise std dev mean')
plt.title('Noise Std Dev parameter over training')
plt.legend()
plt.show()


# In[32]:


# Plot value of bias std dev over training
plt.figure()
plt.plot(varational_regressor.noise_stds, label='fit')
plt.xlabel('Training Step')
plt.ylabel('Noise std dev scale')
plt.title('Noise Std Dev scale parameter over training')
plt.legend()
plt.show()


# In[33]:


# Plot mean squared error over training
plt.figure()
plt.plot(varational_regressor.mses[1:])
plt.xlabel('Training Step')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error over training')
plt.show()


# In[34]:


# Plot ELBO loss over training
plt.figure()
plt.plot(varational_regressor.losses)
plt.xlabel('Training Step')
plt.ylabel('ELBO Loss')
plt.title('ELBO loss over training')
plt.show()


# ### Posterior
# 
# Like with MCMC sampling, we can sample from the model using the variational distributions at the end of training.  This was done above when we ran the training session (see the code block `Draw samples from the posterior`).  Now we'll plot the computed posteriors (distributions below) against the true parameter values (vertical lines).

# In[35]:


def post_plot(data, title='', ax=None, true=None, prc=95):
  if ax is None:
    ax = plt.gca()
  sns.distplot(data, ax=ax)
  ax.axvline(x=np.percentile(data, (100-prc)/2), linestyle='--')
  ax.axvline(x=np.percentile(data, 100-(100-prc)/2), linestyle='--')
  ax.title.set_text(title+" distribution")
  if true is not None:
    ax.axvline(x=true)

# Plot weight posteriors
fig, axes = plt.subplots(data.number_of_dimensions+2, 1, sharex=True)
fig.set_size_inches(6.4, 8)
for i in range(data.number_of_dimensions):
  post_plot(w_post[:,i,0], title='W[{}]'.format(i), 
            ax=axes[i], true=data.w_true[i])
  
# Plot Bias posterior
post_plot(b_post[:,0], title='Bias', 
          ax=axes[data.number_of_dimensions], true=data.b_true)

# Plot noise std dev posterior
post_plot(n_post[:,0], title='Noise std',
          ax=axes[data.number_of_dimensions+1], true=data.noise_std_true)
  
plt.show()


# ### Predictive Distributions
# 
# Variational inference can also get us the posterior predictive distributions.  Again these were computed while running the training session above (see the code block `Draw predictive distribution samples`).  Now we can compare the posterior predictive distributions for each validation datapoint (distributions below) to the true $y$ value of that validation datapoint (vertical lines).

# In[36]:


# Plot random datapoints and their prediction distributions
fig, axes = plt.subplots(4, 2, sharex='all', sharey='all')
for i in range(4):
  for j in range(2):
    ix = i*2+j
    pred_dist = prediction_dist_var[:, ix, 0]
    sns.kdeplot(pred_dist, shade=True, ax=axes[i][j])
    axes[i][j].axvline(x=data.y_val[ix,0])


# Using these posterior predictive distributions, we can compute the coverage of the 95% interval (how often the true $y$ value falls within our 95% confidence interval):

# In[37]:


# Compute proportion of estimates on validation data 
# which fall within the 95% prediction interval
q0 = 2.5
q1 = 97.5
within_conf_int = np.zeros(data.N_val)
for i in range(data.N_val):
  pred_dist = prediction_dist_var[:, i, 0]
  p0 = np.percentile(pred_dist, q0)
  p1 = np.percentile(pred_dist, q1)
  if p0<=data.y_val[i] and p1>data.y_val[i]:
    within_conf_int[i] = 1

print('%0.1f %% of validation samples are w/i the %0.1f %% prediction interval' 
      % (100*np.mean(within_conf_int), q1-q0))


# ## Comparing MCMC and Variational Fits
# 
# Now we can compare the fit using MCMC to the variational fit.  We'll first see how the parameter posteriors from each method stack up to each other, and then we'll compare the posterior predictive distributions from each method.

# ### Posteriors
# 
# The blue distributions below are the posterior distributions for each parameter as estimated by MCMC, while the green distributions are the posterior distributions as estimated by variational inference.  For the noise standard deviation parameter, this is a point estimate for the variational model, and so that estimate is a vertical line.  The dotted black vertical lines are the true parameter values used to generate the data.

# In[38]:


# Plot chains and distributions for coefficients
fig, axes = plt.subplots(data.number_of_dimensions+2, 1, sharex='all')
fig.set_size_inches(6.4, 8)
for i in range(data.number_of_dimensions):
  t_ax = axes[i]
 # sns.kdeplot(coeffs_samples[:,i], #MCMC posterior
 #             ax=t_ax, label='MCMC')
  sns.kdeplot(w_post[:,i,0], #variational posterior
              ax=t_ax, label='Variational') 
  t_ax.axvline(x=data.w_true[i], #true value
               color='k', linestyle=':') 
  t_ax.title.set_text('W[{}]'.format(i))
  
# Plot chains and distributions for bias
t_ax = axes[data.number_of_dimensions]
#sns.kdeplot(bias_samples[:,0], #MCMC posterior
 #           ax=t_ax, label='MCMC')
sns.kdeplot(b_post[:,0], #variational posterior
            ax=t_ax, label='Variational') 
t_ax.axvline(x=data.b_true, #true value
             color='k', linestyle=':') 
t_ax.title.set_text('Bias')

# Plot chains and distributions for noise std dev
t_ax = axes[data.number_of_dimensions+1]
#sns.kdeplot(noise_std_samples[:,0], #MCMC posterior
#            ax=t_ax, label='MCMC')
sns.kdeplot(n_post[:,0], #variational estimate
            ax=t_ax, label='Variational') 
t_ax.axvline(x=data.noise_std_true, #true value
             color='k', linestyle=':') 
t_ax.title.set_text('Noise std dev')

axes[data.number_of_dimensions+1].set_xlabel("Parameter value")
fig.tight_layout()
plt.show()


# The posteriors for the four weight parameters are very similar for both methods!  The bias posterior of the variational model is a lot sharper than the posterior for the same parameter using MCMC sampling.  I'm not entirely sure why that is - if anyone has any ideas please comment! 
# 
# Finally, the noise standard deviation parameter's posterior as computed by variational inference is similar to the posterior obtained via MCMC, but not exactly the same.  The distributions appear to have similar means and variances, but notice that the posterior computed with MCMC is slightly more positively skewed.  The variational posterior has a non-skewed normal distribution shape because, remember, we forced each parameter's posterior to be a normal distribution when we replaced each parameter with a normal distribution in order to do variational inference!

# ### Predictive Distributions 
# 
# Let's also compare the posterior predictive distributions on some individual validation datapoints.

# In[39]:


# Plot some datapoints and their prediction distributions
# as computed by MCMC and variational Bayes
fig, axes = plt.subplots(4, 2, sharex='all')
for i in range(4):
  for j in range(2):
    ix = i*2+j
    pred_dist_var = prediction_dist_var[:, ix, 0]
    #pred_dist = prediction_distribution[ix,:]
    #sns.kdeplot(pred_dist, ax=axes[i][j])
    sns.kdeplot(pred_dist_var, ax=axes[i][j])
    axes[i][j].axvline(x=data.y_val[ix,0], color='k')


# Comparing the coverage of the 95% interval between the two inference methods, it looks like MCMC's uncertainty estimates are a bit too high, and the variational uncertainty estimates are a bit too low.

# In[40]:


# Compute proportion of estimates on validation data 
# which fall within the 95% prediction interval
q0 = 2.5
q1 = 97.5
within_conf_int = np.zeros(data.N_val)
within_conf_int_var = np.zeros(data.N_val)
for i in range(data.N_val):
  # For MCMC
  #pred_dist = prediction_distribution[i,:]
  #p0 = np.percentile(pred_dist, q0)
  #p1 = np.percentile(pred_dist, q1)
  #if p0<=y_val[i] and p1>y_val[i]:
  #  within_conf_int[i] = 1
    
  # For variational model
  pred_dist_var = prediction_dist_var[:, i, 0]
  p0 = np.percentile(pred_dist_var, q0)
  p1 = np.percentile(pred_dist_var, q1)
  if p0<=data.y_val[i] and p1>data.y_val[i]:
    within_conf_int_var[i] = 1
    
# Print 95% interval coverage
print('Percent of validation samples w/i the %0.1f %% confidence interval:' 
      % (q1-q0))
#print('MCMC: %0.1f%%' % (100*np.mean(within_conf_int)))
print('Variational: %0.1f%%' % (100*np.mean(within_conf_int_var)))

