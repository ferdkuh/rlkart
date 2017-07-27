import time
import multiprocessing
import threading
import sys
import numpy as np

from mariokart import MarioKartEnv

from PIL import Image
import scipy.misc


from myrl.learner_manager import LearnerProcessManager
from myrl.network import Network

ROM_PATH = r"../res/Mario Kart 64 (U) [!].z64"
def create_environment(index):
	return MarioKartEnv(ROM_PATH, index)

NUM_LEARNERS = 1
STATE_SHAPE = [84, 84]
BATCH_SIZE = 30

GAMMA = 0.99

# https://github.com/Alfredvc/paac/blob/master/paac.py
def sample_action_from_policy(probabilities):
	probabilities -= np.finfo(np.float32).epsneg
	action_indices = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probabilities]
	return action_indices

def signal_handler(signal, frame):
	print("aborting due to ctrl-c")
	manager.stop_learners()
	sys.exit(0)

if __name__ == "__main__":

	import signal, os
	import tensorflow as tf

	# try to handle ctrl-c gracefully
	signal.signal(signal.SIGINT, signal_handler)
	session = tf.InteractiveSession()	

	manager = LearnerProcessManager(NUM_LEARNERS, create_environment)
	network = Network(num_actions = 4, state_shape = STATE_SHAPE)
	
	saver = tf.train.Saver()
	

	# settings for the optimizer and gradient clipping are as in https://github.com/Alfredvc/paac/
	optimizer = tf.train.RMSPropOptimizer(0.0224, decay=0.99, epsilon=0.1)
	gradients, variables = zip(*optimizer.compute_gradients(network.loss))
	gradients, _ = tf.clip_by_global_norm(gradients, 3.0)
	minimize_op = optimizer.apply_gradients(zip(gradients, variables))	

	session.run(tf.global_variables_initializer())
	saver.restore(session, "checkpoints/380-it.ckpt")

	manager.start_learners()

	# apply on no-op to all environments for initalization
	manager.update_learners([0]*NUM_LEARNERS)

	# setup buffers for all relevant variables
	# since every episode always has the same length, their size can be constant
	states = np.zeros([BATCH_SIZE, NUM_LEARNERS] + STATE_SHAPE)
	rewards = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	values = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	returns = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	actions = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	advantages = np.zeros([BATCH_SIZE, NUM_LEARNERS])

	# some rudimentary logging
	all_episode_rewards = []

	# train for one epsiode with num_iterations steps
	def train_episode(num_iterations=100):
		manager.start_new_episode()
		total_episode_reward = 0.0
		for i in range(num_iterations):

			tic = time.time()

			for t in range(0, BATCH_SIZE):
				# get the current state from shared memory
				current_state = np.copy(manager.shared_memory.states)

				# get next actions from tf
				ops = [network.policy_out, network.value_out]
				feed_dict = { network.states: current_state }
				policy_out, value_out = session.run(ops, feed_dict)
				action_indices = sample_action_from_policy(policy_out)

				# update the environments and wait for results
				manager.update_learners(action_indices)

				# copy states and rewards into the corresponding buffers				
				states[t] = current_state
				rewards[t] = np.copy(manager.shared_memory.rewards)
				values[t] = value_out
				actions[t] = action_indices

			# bootstrap R from value function for the last seen state
			current_state = current_state = np.copy(manager.shared_memory.states)
			next_state_value = session.run(
				network.value_out,
				feed_dict = { network.states: current_state})

			# compute the estimated returns R
			estimated_return = np.copy(next_state_value)
			for t in reversed(range(BATCH_SIZE)):
				estimated_return = rewards[t] + GAMMA * estimated_return
				returns[t] = np.copy(estimated_return)
				advantages[t] = estimated_return - values[t]

			# flatten all the buffers for training the nn
			batch_states = states.reshape([-1] + STATE_SHAPE)
			batch_advantages = advantages.reshape([-1])
			batch_returns = returns.reshape([-1])
			batch_actions = actions.reshape([-1])

			feed_dict = {
				network.states: batch_states,
				network.action_indices: batch_actions,
				network.returns: batch_returns,
				network.advantages: batch_advantages
			}

			ops = [minimize_op]
			_ = session.run(ops, feed_dict = feed_dict)

			# print some numbers
			log_row = "{:.6f}\t{}"
			total_episode_reward += np.sum(rewards)			
			action_counts = np.bincount(batch_actions.astype(np.uint8))
			mean_reward = np.mean(rewards)
			log_msg = log_row.format(mean_reward, action_counts)
			print(log_msg)

		print("total episode reward: {:.4f}".format(total_episode_reward))
		all_episode_rewards.append(total_episode_reward)

	def play(n=100):
		manager.start_new_episode()
		for i in range(n):
			current_state = np.copy(manager.shared_memory.states)
			ops = [network.policy_out]
			feed_dict = { network.states: current_state }
			policy_out = session.run(ops, feed_dict)[0]
			action_indices = sample_action_from_policy(policy_out)
			manager.update_learners(action_indices, action="generate_frame")