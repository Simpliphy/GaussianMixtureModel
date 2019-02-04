from Timer import Timer

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

class BayesianLinearRegressor_TF_MCMC(object):

    def __init__(self, hmc_parameters):

        self.hmc_parameters = hmc_parameters
        self.log_joint = None
        self.define_joint_posterior_distribution()


        self.coeffs_ = None
        self.bias_  = None
        self.noise_std_  = None
        self.is_accepted_  = None

    def define_joint_posterior_distribution(self):

        def linear_regression(features):
            D = features.shape[1]  # number of dimensions
            coeffs = ed.Normal(  # normal prior on weights
                loc=tf.zeros([D, 1]),
                scale=tf.ones([D, 1]),
                name="coeffs")
            bias = ed.Normal(  # normal prior on bias
                loc=tf.zeros([1]),
                scale=tf.ones([1]),
                name="bias")
            noise_std = ed.HalfNormal(  # half-normal prior on noise std
                scale=tf.ones([1]),
                name="noise_std")
            predictions = ed.Normal(  # normally-distributed noise around predicted values
                loc=tf.matmul(features, coeffs) + bias,
                scale=noise_std,
                name="predictions")
            return predictions

        self.log_joint = ed.make_log_joint_fn(linear_regression)



    def sample_posterior(self, data):

        def target_log_prob_fn(coeffs, bias, noise_std):

            return self.log_joint(
                features=data.x,
                coeffs=coeffs,
                bias=bias,
                noise_std=noise_std,
                predictions=data.y)

        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=self.hmc_parameters.step_size,
            num_leapfrog_steps=self.hmc_parameters.num_leapfrog_steps)

        # Define the chain states
        states, kernel_results = tfp.mcmc.sample_chain(
            num_results=self.hmc_parameters.num_results,
            num_burnin_steps=self.hmc_parameters.n_burnin,
            kernel=kernel,
            current_state=[
                tf.zeros(self.hmc_parameters.coeffs_size, name='init_coeffs'),
                tf.zeros(self.hmc_parameters.bias_size, name='init_bias'),
                tf.ones(self.hmc_parameters.noise_std_size, name='init_noise_std'),
            ])
        coeffs, bias, noise_std = states

        with Timer(), tf.Session() as sess:
            [
                self.coeffs_, self.bias_,self.noise_std_,
                self.is_accepted_,
            ] = sess.run([
                coeffs,
                bias,
                noise_std,
                kernel_results.is_accepted,
            ])

    def return_samples(self):

        coeffs_samples = self.coeffs_[self.hmc_parameters.n_burnin:, :, 0]
        bias_samples = self.bias_[self.hmc_parameters.n_burnin:]
        noise_std_samples = self.noise_std_[self.hmc_parameters.n_burnin:]
        accepted_samples = self.is_accepted_[self.hmc_parameters.n_burnin:]

        return coeffs_samples, bias_samples, noise_std_samples, accepted_samples
