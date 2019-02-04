import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Timer import Timer

tfd = tfp.distributions
train_writer = tf.summary.FileWriter('./train')

class VariationalLinearBayesianRegressor(object):

    def __init__(self, data, variational_parameters):

        tf.reset_default_graph()

        self.data = data
        self.variational_parameters = variational_parameters

        self.weight_mean = None
        self.weight_std = None
        self.bias_mean = None
        self.bias_std = None
        self.noise_std_mean = None
        self.noise_std_std = None
        self.weight_means = None
        self.weight_stds = None
        self.bias_means = None
        self.bias_stds = None
        self.noise_stds = None
        self.noise_means = None
        self.mses = None
        self.losses = None


        self.x_vals =None
        self.y_vals = None
        self.handle = None
        self.training_iterator = None
        self.validation_iterator = None
        self._initiliaze_pipepile()

        self.layer = None
        self.pred_distribution = None
        self.predictions = None
        self.noise_std = None
        self._initialize_layer()

        self._initiliaze_training_paremeters_for_iterations()
        self._define_optimizater()


    def _initialize_layer(self):

        with tf.name_scope("linear_regression", values=[self.x_vals]):
            self.layer = tfp.layers.DenseFlipout(
                units=1,
                activation=None,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn())
            self.predictions = self.layer(self.x_vals)
            self.noise_std = VariationalParameter('noise_std', [1],
                                             constraint=tf.keras.constraints.NonNeg())
            self.pred_distribution = tfd.Normal(loc=self.predictions,
                                           scale=self.noise_std.sample())
    def _initiliaze_pipepile(self):

        self.x_vals, self.y_vals, self.handle, self.training_iterator, self.validation_iterator = (
            build_input_pipeline(self.data.x,
                                 self.data.y,
                                 self.data.x_val,
                                 self.data.y_val,
                                 self.variational_parameters.batch_size,
                                 self.data.N_val))

    def _define_optimizater(self):

        # Compute the -ELBO as the loss, averaged over the batch size
        neg_log_likelihood = -tf.reduce_mean(self.pred_distribution.log_prob(self.y_vals))
        kl_div = sum(self.layer.losses) / self.data.number_of_datapoints
        self.elbo_loss = neg_log_likelihood + kl_div

        # Mean squared error metric for evaluation
        self.mse, self.mse_update_op = tf.metrics.mean_squared_error(
            labels=self.y_vals, predictions=self.predictions)

        # Use ADAM optimizer w/ -ELBO loss
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.variational_parameters.learning_rate)
            self.train_op = optimizer.minimize(self.elbo_loss)
            tf.summary.scalar("elbo_loss", self.elbo_loss)


    def _initiliaze_training_paremeters_for_iterations(self):

        assert self.layer is not None

        self.weight_mean = self.layer.kernel_posterior.mean()[:, 0]
        for dimension in range(self.weight_mean.shape[0]):
            tf.summary.scalar("weight_mean_%d"%dimension, self.weight_mean[dimension])

        self.weight_std = self.layer.kernel_posterior.stddev()[:, 0]
        self.bias_mean = self.layer.bias_posterior.mean()
        self.bias_std = self.layer.bias_posterior.stddev()
        self.noise_std_mean = self.noise_std.mean()
        self.noise_std_std = self.noise_std.stddev()
        self.weight_means = np.zeros((self.variational_parameters.max_steps, self.data.number_of_dimensions))
        self.weight_stds = np.zeros((self.variational_parameters.max_steps, self.data.number_of_dimensions))
        self.bias_means = np.zeros(self.variational_parameters.max_steps)
        self.bias_stds = np.zeros(self.variational_parameters.max_steps)
        self.noise_stds = np.zeros(self.variational_parameters.max_steps)
        self.noise_means = np.zeros(self.variational_parameters.max_steps)
        self.mses = np.zeros(self.variational_parameters.max_steps)
        self.losses = np.zeros(self.variational_parameters.max_steps)

    def fit(self):
        # Initialization op
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())


        # Run the training session
        with tf.Session() as sess:
            sess.run(init_op)

            merged = tf.summary.merge_all()
            train_writer.add_graph(sess.graph)

            # Training loop
            self.train_handle = sess.run(self.training_iterator.string_handle())
            self.val_handle = sess.run(self.validation_iterator.string_handle())
            with Timer():
                for iS in range(self.variational_parameters.max_steps):
                    [
                        _,
                        _,
                        self.mses[iS],
                        self.losses[iS],
                        self.weight_means[iS, :],
                        self.weight_stds[iS, :],
                        self.bias_means[iS],
                        self.bias_stds[iS],
                        self.noise_means[iS],
                        self.noise_stds[iS],
                        summary
                    ] = sess.run([
                        self.train_op,
                        self.mse_update_op,
                        self.mse,
                        self.elbo_loss,
                        self.weight_mean,
                        self.weight_std,
                        self.bias_mean,
                        self.bias_std,
                        self.noise_std_mean,
                        self.noise_std_std,
                        merged
                    ], feed_dict={self.handle: self.train_handle})

                    train_writer.add_summary(summary, iS)

            Nmc = 1000
            w_draw = self.layer.kernel_posterior.sample(Nmc)
            b_draw = self.layer.bias_posterior.sample(Nmc)
            n_draw = self.noise_std.sample(Nmc)
            w_post, b_post, n_post = sess.run([w_draw, b_draw, n_draw])

            # Draw predictive distribution samples
            prediction_dist_var = sess.run((self.pred_distribution.sample(Nmc)),
                                           feed_dict={self.handle: self.val_handle})





        return w_post, b_post, n_post, prediction_dist_var


def build_input_pipeline(x, y, x_val, y_val, batch_size, N_val):

    '''Build an Iterator switching between train and heldout data.
    Args:
    x: Numpy `array` of training features, indexed by the first dimension.
    y: Numpy `array` of training labels, with the same first dimension as `x`.
    x_val: Numpy `array` of validation features, indexed by the first dimension.
    y_val: Numpy `array` of validation labels, with the same first dimension as `x_val`.
    batch_size: Number of elements in each training batch.
    N_val: Number of examples in the validation dataset
    Returns:
    batch_features: `Tensor` feed  features, of shape
      `[batch_size] + x.shape[1:]`.
    batch_labels: `Tensor` feed of labels, of shape
      `[batch_size] + y.shape[1:]`.
    '''
    # Build an iterator over training batches.
    training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    training_batches = training_dataset.shuffle(
      50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
    train_iterator = training_batches.make_one_shot_iterator()

    # Build a iterator over the validation set with batch_size=N_val,
    # i.e., return the entire heldout set as a constant.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_frozen = val_dataset.take(N_val).repeat().batch(N_val)
    val_iterator = val_frozen.make_one_shot_iterator()

    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
    batch_features, batch_labels = feedable_iterator.get_next()

    return batch_features, batch_labels, handle, train_iterator, val_iterator

def VariationalParameter(name, shape, constraint=None):

  """Generates variational distribution(s)"""
  means = tf.get_variable(name+'_mean',
                          initializer=tf.ones([1]),
                          constraint=constraint)
  stds = tf.get_variable(name+'_std',
                         initializer=-2.3*tf.ones([1]))
  return tfd.Normal(loc=means, scale=tf.math.exp(stds))

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
