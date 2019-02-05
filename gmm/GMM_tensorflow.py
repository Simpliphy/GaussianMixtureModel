import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp

from gmm.GaussianSoftClusteringParameters import GaussianSoftClusteringParameters

np.random.seed(42)

class GaussianSoftClustering(object):
    """
    Based on assignment from week 2 Bayesian method for machine learning of Coursera.
    """

    def __init__(self):

        self.parameters = GaussianSoftClusteringParameters()

    def _E_step(self, observations, parameters):

        """
        Performs E-step on GMM model
        # P(i|x)=p(x|i)p(i)/z
        # p(x_n|i)=N(x_n| mu_i,sigma_i)

        changed:
        --------

        parameters.hidden_states_distribution: [|data| x |states|], probabilities of states for objects

        keeped constant:
        ----------------

        parameters.mu
        parameters.sigma
        parameters.hidden_states_prior

        """


        assert isinstance(observations, np.ndarray)

        number_of_observations = observations.shape[0]
        number_of_clusters = parameters.hidden_states_prior.shape[0]
        with tf.variable_scope("E_step", reuse=tf.AUTO_REUSE):

            hidden_states_distribution_tf = tf.get_variable(name="hidden_state_distribution",
                                                            shape=(number_of_observations, number_of_clusters),
                                                            dtype=tf.float64)


        with tf.Session() as sess:

            init = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

            sess.run(init)


            for cluster_index in range(number_of_clusters):

                multivariate_normal_pdf = tfp.distributions.MultivariateNormalFullCovariance(loc=parameters.mu[cluster_index, :],
                                                                    covariance_matrix=parameters.sigma[cluster_index, ...])

                multivariate_normal_prob = multivariate_normal_pdf.prob(observations)
                prior_of_state = tf.convert_to_tensor(parameters.hidden_states_prior[cluster_index], dtype=tf.float64)
                product_prior_and_normal = multivariate_normal_prob * prior_of_state
                assign = hidden_states_distribution_tf[:, cluster_index].assign(product_prior_and_normal)


                sess.run([multivariate_normal_prob,prior_of_state,product_prior_and_normal,assign])


            hidden_states_distribution_tf_normalization = tf.reduce_sum(hidden_states_distribution_tf, axis=1)
            hidden_states_normalization_constants = tf.expand_dims(hidden_states_distribution_tf_normalization, 0)

            hidden_states_distribution_tf_normalized = hidden_states_distribution_tf / tf.transpose(
                hidden_states_normalization_constants)

            [_, hidden_states] = sess.run([hidden_states_distribution_tf_normalization, hidden_states_distribution_tf_normalized])

            parameters.hidden_states_distribution = hidden_states

    def _M_step(self, observations, parameters):
        """
        Performs M-step on GMM model

        changed:
        --------

        parameters.mu
        parameters.sigma
        parameters.hidden_states_prior

        keeped constant:
        ----------------

        parameters.hidden_states_distribution: [|data| x |states|], probabilities of states for objects

        """

        assert isinstance(observations, np.ndarray)
        assert isinstance(parameters.hidden_states_distribution, np.ndarray)

        number_of_observations = observations.shape[0]
        number_of_clusters = parameters.hidden_states_distribution.shape[1]
        number_of_features = observations.shape[1]

        with tf.variable_scope("M_step", reuse=tf.AUTO_REUSE):

            hidden_states_distribution_tf = tf.convert_to_tensor(parameters.hidden_states_distribution,
                                                                  dtype=tf.float64)

            normalization_constants = tf.expand_dims(tf.reduce_sum(hidden_states_distribution_tf, axis=0), 1)

            mu = tf.matmul(hidden_states_distribution_tf, observations)/normalization_constants

            hidden_states_prior = normalization_constants / number_of_observations


            #sigma = np.zeros((number_of_clusters, number_of_features, number_of_features))

            for state_index in range(number_of_clusters):

                observation_translated = tf.convert_to_tensor(observations) - mu[state_index]

                hidden_state_weights_diag = np.diag(parameters.hidden_states_distribution[:, state_index])

                sigma_state = np.dot(np.dot(x_mu.T, hidden_state_weights_diag), x_mu)
                sigma[state_index, ...] = sigma_state / normalization_constants[state_index]


        parameters.hidden_states_prior = hidden_states_prior
        parameters.mu = mu
        parameters.sigma = sigma

    def compute_vlb(self, observations, parameters):
        """
        observations: [ |data| x |features| ]
        
        hidden_states_distribution: [|data| x |states|]
        states_prior: [|states|]
        
        mu: [|states| x |features|]
        sigma: [|states| x |features| x |features|] 

        Returns value of variational lower bound
        """
        
        assert isinstance(observations, np.ndarray)
        assert isinstance(parameters.hidden_states_prior, np.ndarray)
        assert isinstance(parameters.mu, np.ndarray)
        assert isinstance(parameters.sigma, np.ndarray)
        assert isinstance(parameters.hidden_states_distribution, np.ndarray)
        
        number_of_observations = observations.shape[0]
        number_of_clusters = parameters.hidden_states_distribution.shape[1]

        loss_per_observation = np.zeros(number_of_observations)
        for k in range(number_of_clusters):

            energy = parameters.hidden_states_distribution[:, k] * (np.log(parameters.hidden_states_prior[k]) +
                        multivariate_normal.logpdf(observations, mean=parameters.mu[k, :], cov=parameters.sigma[k, ...]))

            entropy = parameters.hidden_states_distribution[:, k] * np.log(parameters.hidden_states_distribution[:, k])

            loss_per_observation += energy
            loss_per_observation -= entropy

        total_loss = np.sum(loss_per_observation)

        return total_loss

    def train_EM(self, observations, number_of_clusters, reducing_factor=1e-3, max_iter=100, restarts=10):

   
        number_of_features = observations.shape[1] 
        number_of_observations = observations.shape[0]

        best_loss = -1e7
        best_parameters = GaussianSoftClusteringParameters()

        for _ in tqdm(range(restarts)):

            try:
                parameters = GaussianSoftClusteringParameters()
                parameters.initialize_parameters(number_of_clusters,
                                                 number_of_features,
                                                 number_of_observations)

                self._E_step(observations, parameters)

                prev_loss = self.compute_vlb(observations,
                                             parameters)

                for _ in range(max_iter):

                    self._E_step(observations, parameters)
                    self._M_step(observations, parameters)

                    loss = self.compute_vlb(observations, parameters)

                    if loss / prev_loss < reducing_factor:
                        break

                    if loss > best_loss:

                        best_loss = loss
                        best_parameters = parameters

                    prev_loss = loss

            except np.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                pass

        return best_loss, best_parameters
