import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

from GaussianSoftClusteringParameters import GaussianSoftClusteringParameters

np.random.seed(42)

class GaussianSoftClustering(object):
    """
    Based on assignment from week 2 Bayesian method for machine learning of Coursera.
    """

    def __init__(self):

        pass

    def E_step(self, observations, hidden_states_prior, mu, sigma):

        """
        Performs E-step on GMM model
        # P(i|x)=p(x|i)p(i)/z
        # p(x_n|i)=N(x_n| mu_i,sigma_i)

        Returns:
        hidden_states_distribution: [|data| x |states|], probabilities of states for objects
        """
        assert isinstance(observations, np.ndarray)

        number_of_observations = observations.shape[0]
        number_of_clusters = hidden_states_prior.shape[0]
        hidden_states_distribution = np.zeros((number_of_observations, number_of_clusters))

        for cluster_index in range(number_of_clusters):

            multivariate_normal_pdf = multivariate_normal.pdf(observations,
                                                              mean=mu[cluster_index, :],
                                                              cov=sigma[cluster_index, ...])

            hidden_states_distribution[:, cluster_index] = multivariate_normal_pdf * (hidden_states_prior[cluster_index])

        hidden_states_distribution /= np.sum(hidden_states_distribution, 1).reshape(-1, 1)

        return hidden_states_distribution

    def M_step(self, observations, hidden_states_distribution):
        """
        Performs M-step on GMM model
        """

        assert isinstance(observations, np.ndarray)
        assert isinstance(hidden_states_distribution, np.ndarray)

        number_of_objects = observations.shape[0]
        number_of_clusters = hidden_states_distribution.shape[1]
        number_of_features = observations.shape[1]  # dimension of each object

        normalization_constants = np.sum(hidden_states_distribution, 0)  # (K,)

        mu = np.dot(hidden_states_distribution.T, observations) / normalization_constants.reshape(-1, 1)
        hidden_states_prior = normalization_constants / number_of_objects
        sigma = np.zeros((number_of_clusters, number_of_features, number_of_features))

        for cluster_index in range(number_of_clusters):

            x_mu = observations - mu[cluster_index]
            gamma_diag = np.diag(hidden_states_distribution[:, cluster_index])

            sigma_k = np.dot(np.dot(x_mu.T, gamma_diag), x_mu)
            sigma[cluster_index, ...] = sigma_k / normalization_constants[cluster_index]

        return hidden_states_prior, mu, sigma

    def compute_vlb(self, observations, states_prior, mu, sigma, hidden_states_distribution):
        """
        observations: [ |data| x |features| ]
        
        hidden_states_distribution: [|data| x |states|]
        states_prior: [|states|]
        
        mu: [|states| x |features|]
        sigma: [|states| x |features| x |features|] 

        Returns value of variational lower bound
        """
        
        assert isinstance(observations, np.ndarray)
        assert isinstance(states_prior, np.ndarray)
        assert isinstance(mu, np.ndarray)
        assert isinstance(sigma, np.ndarray)
        assert isinstance(hidden_states_distribution, np.ndarray)
        
        number_of_observations = observations.shape[0]
        number_of_clusters = hidden_states_distribution.shape[1]

        loss_per_observation = np.zeros(number_of_observations)
        for k in range(number_of_clusters):

            energy = hidden_states_distribution[:, k] * (np.log(states_prior[k]) + multivariate_normal.logpdf(observations, mean=mu[k, :], cov=sigma[k, ...]))
            entropy = hidden_states_distribution[:, k] * np.log(hidden_states_distribution[:, k])

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
                parameters.initialize_parameters( number_of_clusters, number_of_features, number_of_observations)

                parameters.hidden_states_distribution = self.E_step(observations, parameters.hidden_states_prior, parameters.mu, parameters.sigma)

                prev_loss = self.compute_vlb(observations,
                                             parameters.hidden_states_prior,
                                             parameters.mu,
                                             parameters.sigma,
                                             parameters.hidden_states_distribution)

                for _ in range(max_iter):

                    gamma = self.E_step(observations, parameters.hidden_states_prior, parameters.mu, parameters.sigma)
                    parameters.hidden_states_prior, parameters.mu, parameters.sigma = self.M_step(observations, gamma)

                    loss = self.compute_vlb(observations,
                                            parameters.hidden_states_prior,
                                            parameters.mu,
                                            parameters.sigma,
                                            parameters.hidden_states_distribution)

                    if loss / prev_loss < reducing_factor:
                        break

                    if loss > best_loss:

                        best_loss = loss
                        best_parameters = parameters

                    prev_loss = loss

            except np.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                pass

        return best_loss, best_parameters.hidden_states_prior, best_parameters.mu, best_parameters.sigma, best_parameters.hidden_states_distribution

