import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected, flatten

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

BETA = 0.01

class Network():

	def __init__(self, num_actions, state_shape):

		self.states = tf.placeholder(tf.float32, shape = [None] + state_shape, name='input_states')
		inputs = tf.expand_dims(self.states, axis=-1)

		# basic conv net - how should they be initialized?
		self.conv1 = conv2d(
			inputs, num_outputs = 16, 
			kernel_size = 8, stride = 4, 
			activation_fn = tf.nn.elu, padding='same')

		self.conv2 = conv2d(
			self.conv1, num_outputs = 32, 
			kernel_size = 4, stride = 2, 
			activation_fn = tf.nn.elu, padding='same')

		self.conv3 = conv2d(
			self.conv2, num_outputs = 64, 
			kernel_size = 3, stride = 1,
			 activation_fn = tf.nn.elu, padding='same')

		conv3_flat = flatten(self.conv3)

		self.fc = fully_connected(
			conv3_flat, num_outputs = 512,
			activation_fn = tf.nn.relu)

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

		# loss ops
		self.action_indices = tf.placeholder(tf.uint8, [None], name='action_indices')		
		self.returns = tf.placeholder(tf.float32, [None], name='discounted_returns')
		self.advantages = tf.placeholder(tf.float32, [None], name='advantages')
		# self.values = tf.placeholder(tf.float32, [None], name='estimated_values')

		# what is the reason everybody computes advantages in python?
		# advantages = self.returns - self.values

		actions_one_hot = tf.one_hot(self.action_indices, num_actions)

		log_prob_tf = tf.nn.log_softmax(policy_logits)
		prob_tf = self.policy_out

		pi_loss = -tf.reduce_sum(tf.reduce_sum(log_prob_tf * actions_one_hot, [1]) * self.advantages)
		vf_loss = 0.5 * tf.reduce_sum(tf.squared_difference(self.value_out, self.returns))
		entropy = -tf.reduce_sum(prob_tf * log_prob_tf)
		loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

		# log_policy = tf.nn.log_softmax(policy_logits)

		# responsible_outputs = tf.reduce_sum(log_policy * actions_one_hot, axis=1)
		# policy_loss = tf.reduce_sum(responsible_outputs * entropy)

		# advantages = -tf.reduce_sum(self.policy_out * log_policy)
		# entropy_sum = tf.reduce_sum(entropy)

		# value_loss = tf.reduce_sum(tf.squared_difference(self.returns, self.values))

		# value_loss = 0.5 * tf.reduce_sum(tf.square(self.returns - self.values))
		# policy_entropy = -tf.reduce_sum(self.policy_out * log_policy)
		# policy_loss = -tf.reduce_sum(responsible_outputs * advantages)
		# loss = 0.5 * value_loss + policy_loss - policy_entropy * BETA

		self.policy_loss = pi_loss
		self.value_loss = vf_loss
		self.loss = loss

		# policy_s = tf.reduce_sum(log_policy * actions_one_hot, axis=1)	# compute value of policy vector for chosen action
		# policy_entropy = -tf.reduce_sum(self.policy_out * log_policy, axis=1)

		# #self.policy_loss = -tf.reduce_sum(advantages * policy_s) + BETA * policy_entropy
		# self.policy_loss = tf.reduce_mean(-1.0 * (advantages * policy_s + BETA * policy_entropy))
		# self.value_loss = tf.reduce_mean(tf.square(self.returns - self.values))
		# self.loss = self.policy_loss + 0.25 * self.value_loss



