class HMC_parameters(object):

    def __init__(self, number_of_dimensions):

        self.num_results = int(10e3)  # number of hmc iterations
        self.n_burnin = int(5e3)  # number of burn-in steps
        self.step_size = 0.01
        self.num_leapfrog_steps = 10

        # Parameter sizes
        self.coeffs_size = [number_of_dimensions, 1]
        self.bias_size = [1]
        self.noise_std_size = [1]
        self.number_of_dimensions = number_of_dimensions

