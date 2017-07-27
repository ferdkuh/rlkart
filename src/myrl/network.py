import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected, flatten
from tensorflow.python.framework import ops

# https://github.com/openai/universe-starter-agent/blob/master/model.py
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
def selu(x):
    with ops.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

class Network():

	def __init__(self, num_actions, state_shape):

		self.states = tf.placeholder(tf.float32, shape = [None] + state_shape, name='input_states')
		inputs = tf.expand_dims(self.states, axis=-1)

		# basic conv net, as described in nature DQN paper, but just for fun with SELU units
		self.conv1 = conv2d(
			inputs, num_outputs = 16, 
			kernel_size = 8, stride = 4, 
			activation_fn = selu, padding='same')

		self.conv2 = conv2d(
			self.conv1, num_outputs = 32, 
			kernel_size = 4, stride = 2, 
			activation_fn = selu, padding='same')

		self.conv3 = conv2d(
			self.conv2, num_outputs = 64, 
			kernel_size = 3, stride = 1,
			 activation_fn = selu, padding='same')

		conv3_flat = flatten(self.conv3)

		self.fc = fully_connected(
			conv3_flat, num_outputs = 512,
			activation_fn = selu)

		# output layers
		policy_logits = fully_connected(
			inputs = self.fc,
			num_outputs = num_actions,
			activation_fn=None,
			weights_initializer=normalized_columns_initializer(0.01), 
			biases_initializer=None)

		value_out = fully_connected(
			inputs = self.fc,
			num_outputs = 1,
			activation_fn=None,
			weights_initializer=normalized_columns_initializer(0.01), 
			biases_initializer=None)

		self.value_out = tf.squeeze(value_out)
		self.policy_out = tf.nn.softmax(policy_logits)

		### loss computation

		# additional inputs for loss function
		self.action_indices = tf.placeholder(tf.uint8, [None], name='action_indices')		
		self.returns = tf.placeholder(tf.float32, [None], name='discounted_returns')
		self.advantages = tf.placeholder(tf.float32, [None], name='advantages')

		actions_one_hot = tf.one_hot(self.action_indices, num_actions)
		log_prob_tf = tf.nn.log_softmax(policy_logits)
		prob_tf = self.policy_out

		# the loss calculation is taken directly from https://github.com/Alfredvc/paac
		output_layer_entropy = tf.reduce_sum(tf.multiply(tf.constant(-1.0), tf.multiply(prob_tf, log_prob_tf)), reduction_indices=1)
		critic_loss = tf.subtract(self.returns, self.value_out)
		log_output_selected_action = tf.reduce_sum(tf.multiply(log_prob_tf, actions_one_hot), reduction_indices=1)
		actor_objective_advantage_term = tf.multiply(log_output_selected_action, self.advantages)
		actor_objective_entropy_term = tf.multiply(0.005, output_layer_entropy)
		actor_objective_mean = tf.reduce_mean(tf.multiply(
			tf.constant(-1.0), 
			tf.add(actor_objective_advantage_term, actor_objective_entropy_term)),
			name='mean_actor_objective')
		critic_loss_mean = tf.reduce_mean(tf.scalar_mul(0.25, tf.pow(critic_loss, 2)), name='mean_critic_loss')
		loss = tf.scalar_mul(tf.constant(5.0), actor_objective_mean + critic_loss_mean)
		
		self.entropy = output_layer_entropy
		self.policy_loss = actor_objective_mean
		self.value_loss = critic_loss_mean
		self.loss = loss