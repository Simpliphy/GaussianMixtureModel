import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp

from gmm.GaussianSoftClusteringParameters import GaussianSoftClusteringParameters
import functools

def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

np.random.seed(42)

class GaussianSoftClustering(object):
    """
    Based on assignment from week 2 Bayesian method for machine learning of Coursera.
    """

    def __init__(self, observations, number_of_states=3):

        self.parameters = GaussianSoftClusteringParameters()

        self.data = observations
        self.number_of_states = number_of_states
        self.number_of_observations = observations.shape[0]
        self.number_of_features = observations.shape[1]

        self.training_parameters = ParametersGMM_TF(self.number_of_states,
                                                     self.number_of_features,
                                                     self.number_of_observations)

        self._M_step = None
        self._E_step = None


    @property
    def E_step(self):

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

        #hidden_states_distribution_tf = tf.get_variable(name="hidden_state_distribution",
        #                                                    shape=(number_of_observations, number_of_clusters),
        #                                                    dtype=tf.float64)

        list_op = list()
        for cluster_index in range(self.number_of_states):

            multivariate_normal_pdf = tfp.distributions.MultivariateNormalFullCovariance(loc=self.training_parameters.mu[cluster_index, :],
                                                                covariance_matrix=self.training_parameters.sigma[cluster_index, ...])

            multivariate_normal_prob = multivariate_normal_pdf.prob(self.data)
            prior_of_state = self.training_parameters.hidden_states_prior[cluster_index]
            product_prior_and_normal = multivariate_normal_prob * prior_of_state
            list_op.append(self.training_parameters.hidden_states_distribution[:, cluster_index].assign(product_prior_and_normal))

        hidden_states_distribution_tf_normalization = tf.reduce_sum(self.training_parameters.hidden_states_distribution, axis=1)
        hidden_states_normalization_constants = tf.expand_dims(hidden_states_distribution_tf_normalization, 0)

        hidden_states_distribution_tf_normalized = self.training_parameters.hidden_states_distribution / tf.transpose(
            hidden_states_normalization_constants)

        list_op.append(self.training_parameters.hidden_states_distribution.assign(hidden_states_distribution_tf_normalized))
        self._E_step = list_op

        return self._E_step

    def _M_step(self, observations, parameters):
        """
        Performs M-step on GMM model

        changed:
        --------

        parameters.mu     [|states| x |features|]
        parameters.sigma [|states| x |features| x |features|]
        parameters.hidden_states_prior [|states|]

        keeped constant:
        ----------------

        parameters.hidden_states_distribution: [|data| x |states|], probabilities of states for objects

        """

        assert isinstance(observations, np.ndarray)
        assert isinstance(parameters.hidden_states_distribution, np.ndarray)

        number_of_observations = observations.shape[0]
        number_of_clusters = parameters.hidden_states_distribution.shape[1]
        number_of_features = observations.shape[1]

        sigma = np.zeros((number_of_clusters, number_of_features, number_of_features))
        assert number_of_features == 1, "The tensorflow implementation only work for one feature"

        with tf.Session() as sess:

            init = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

            sess.run(init)

            hidden_states_distribution_tf = tf.convert_to_tensor(parameters.hidden_states_distribution,
                                                                  dtype=tf.float64)

            normalization_constants = tf.expand_dims(tf.reduce_sum(hidden_states_distribution_tf, axis=0), 1)

            mu_tf = tf.matmul(tf.transpose(hidden_states_distribution_tf), observations)/normalization_constants
            hidden_states_prior_tf = normalization_constants / number_of_observations

            [mu, hidden_states_prior] = sess.run([mu_tf, hidden_states_prior_tf])

            for state_index in range(number_of_clusters):

                observation_translated = tf.convert_to_tensor(observations) - mu[state_index]
                hidden_state_weights = tf.convert_to_tensor(parameters.hidden_states_distribution[:, state_index])

                observation_translated_squared = tf.square(observation_translated)
                weighted_squared = tf.multiply(observation_translated_squared, tf.expand_dims(hidden_state_weights,1))
                sigma_state = tf.reduce_sum(weighted_squared,axis=0)
                normalize_sigma = sigma_state / normalization_constants[state_index]

                sigma[state_index, ...] = sess.run(normalize_sigma)

        parameters.hidden_states_prior.assign(hidden_states_prior)
        parameters.mu.assign(mu)
        parameters.sigma.assign(sigma)

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
        #assert isinstance(parameters.hidden_states_prior, np.ndarray)
        #assert isinstance(parameters.mu, np.ndarray)
        #assert isinstance(parameters.sigma, np.ndarray)
        #assert isinstance(parameters.hidden_states_distribution, np.ndarray)
        
        #number_of_observations = observations.shape[0]
        number_of_clusters = parameters.hidden_states_distribution.shape[1]

        with tf.Session() as sess:

            total_loss = 0.0

            for k in range(number_of_clusters):

                hidden_states_weights = parameters.hidden_states_distribution[:, k]
                log_hidden_states_prior = tf.log(tf.convert_to_tensor(parameters.hidden_states_prior[k]))

                multivarate_normal_tf = tfp.distributions.MultivariateNormalFullCovariance(loc=parameters.mu[k, :],
                                                                   covariance_matrix=parameters.sigma[k, ...])

                energy = tf.multiply(hidden_states_weights,log_hidden_states_prior) +\
                            multivarate_normal_tf.log_prob(observations)

                entropy = tf.multiply(hidden_states_weights,tf.log(hidden_states_weights))

                loss_per_observation = energy - entropy
                reduce_loss = tf.reduce_sum(loss_per_observation, axis=0)
                loss_hidden_state = sess.run(reduce_loss)
                total_loss += loss_hidden_state

        return total_loss

    def train_EM(self, observations, reducing_factor=1e-3, max_iter=100, restarts=10):
   
       # number_of_features = observations.shape[1]
        #number_of_observations = observations.shape[0]
        #number_of_clusters = self.parameters.number_of_clusters

        #best_loss = -1e7
        #best_parameters = ParametersGMM("best", *self.parameters.dimensions)
        #best_parameters.initialize_parameters()

        for _ in tqdm(range(restarts)):

            try:
                #parameters = ParametersGMM("calculation", *self.parameters.dimensions)
                #parameters.initialize_parameters()

                sess = tf.Session()
                init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
                sess.run(init)
                sess.run(self.E_step)

               # prev_loss = self.compute_vlb(observations, parameters)

                for _ in range(max_iter):

                    sess.run(self.E_step)
                    # self._M_step(observations, parameters)

                   # loss = self.compute_vlb(observations, parameters)

                   # if loss / prev_loss < reducing_factor:
                   #     break

                   # if loss > best_loss:

                     #   best_loss = loss
                     #   best_parameters = parameters

                  #  prev_loss = loss

            except np.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                pass

        return best_loss, best_parameters


class ParametersGMM_TF():

    def __init__(self, number_of_clusters, number_of_features, number_of_observations):

        self.number_of_clusters = number_of_clusters
        self.number_of_features = number_of_features
        self.number_of_observations = number_of_observations

        self.hidden_states_distribution = None
        self.hidden_states_prior = None
        self.mu = None
        self.sigma = None

        initial_prior = tf.constant(1 / float(self.number_of_clusters) * np.ones(self.number_of_clusters))
        self.hidden_states_prior = tf.get_variable(name="hidden_states_prior",
                                                   initializer=initial_prior,
                                                   # shape=(self.number_of_clusters,1),
                                                   )  # dtype=tf.float64)

        initial_mu = tf.constant(np.random.randn(self.number_of_clusters, self.number_of_features))
        self.mu = tf.get_variable(name="mu",
                                  initializer=initial_mu,
                                  # shape=(self.number_of_clusters, self.number_of_features),
                                  )  # dtype=tf.float64)

        sigma = np.zeros((self.number_of_clusters, self.number_of_features, self.number_of_features))
        sigma[...] = np.identity(self.number_of_features)
        initial_sigma = tf.constant(sigma)
        self.sigma = tf.get_variable(name="sigma",
                                     initializer=initial_sigma,
                                     # shape=(self.number_of_observations, self.number_of_clusters),
                                     )  # dtype=tf.float64)

        initial_hidden_states_distributions = tf.constant(
            np.zeros((self.number_of_observations, self.number_of_clusters)))
        self.hidden_states_distribution = tf.get_variable(name="hidden_state_distribution",
                                                          initializer=initial_hidden_states_distributions,
                                                          # shape=(self.number_of_observations, self.number_of_clusters),
                                                          )  # dtype=tf.float64)

        # with tf.Session() as sess:
        # RUN INIT
        #    sess.run(tf.global_variables_initializer())

    @property
    def dimensions(self):
        return [self.number_of_clusters, self.number_of_features, self.number_of_observations]




