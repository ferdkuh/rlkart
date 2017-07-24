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

NUM_LEARNERS = 12
STATE_SHAPE = [84, 84]
BATCH_SIZE = 30

GAMMA = 0.99

def sample_action_from_policy(probabilities):
	# Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    # as seen in: https://github.com/Alfredvc/paac/blob/master/paac.py
	probabilities -= np.finfo(np.float32).epsneg
	action_indices = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probabilities]
	return action_indices

if __name__ == "__main__":

	import signal, os
	import tensorflow as tf

	def signal_handler(signal, frame):
		print("aborting due to ctrl-c")
		manager.stop_learners()

	signal.signal(signal.SIGINT, signal_handler)

	manager = LearnerProcessManager(NUM_LEARNERS, create_environment)
	network = Network(num_actions = 9, state_shape = STATE_SHAPE)

	session = tf.InteractiveSession()
	saver = tf.train.Saver()
	#optimizer = tf.train.RMSPropOptimizer(0.0224, decay=0.99, epsilon=0.1)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01, epsilon=0.1)
	gradients, variables = zip(*optimizer.compute_gradients(network.loss))
	gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
	minimize_op = optimizer.apply_gradients(zip(gradients, variables))
	#minimize_op = optimizer.minimize(network.loss)

	# minimize_op = optimizer.minimize(network.loss)

	# this blocks until each learner is ready
	session.run(tf.global_variables_initializer())
	saver.restore(session, "checkpoints/last.ckpt")

	manager.start_learners()
	# hack: do one no-op, so the learners will put something in their shared state array
	manager.update_learners([0]*NUM_LEARNERS)

	states = np.zeros([BATCH_SIZE, NUM_LEARNERS] + STATE_SHAPE)
	rewards = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	values = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	returns = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	actions = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	advantages = np.zeros([BATCH_SIZE, NUM_LEARNERS])

	global_steps = 0
	all_episode_rewards = []

	def train_a_little(num_iterations=100):

		total_episode_reward = 0.0
		for i in range(num_iterations):

			tic = time.time()

			if i % 500 == 0:
				print("starting new episode")
				manager.start_new_episode()

			for t in range(0, BATCH_SIZE):

				current_state = np.copy(manager.shared_memory.states)

				# get next actions from tf
				ops = [network.policy_out, network.value_out]
				feed_dict = { network.states: current_state }
				policy_out, value_out = session.run(ops, feed_dict)
				action_indices = sample_action_from_policy(policy_out)
				#action_indices = np.array([3]*NUM_LEARNERS, dtype=np.uint8)

				# update the environments and wait for results
				manager.update_learners(action_indices)

				# copy states and rewards into the corresponding buffers				
				states[t] = current_state
				rewards[t] = np.copy(manager.shared_memory.rewards)
				values[t] = value_out
				actions[t] = action_indices

			# bootstrap R from value function
			current_state = current_state = np.copy(manager.shared_memory.states)
			next_state_value = session.run(
				network.value_out,
				feed_dict = { network.states: current_state})

			estimated_return = np.copy(next_state_value)

			for t in reversed(range(BATCH_SIZE)):
				estimated_return = rewards[t] + GAMMA * estimated_return
				returns[t] = np.copy(estimated_return)
				advantages[t] = estimated_return - values[t]

			# train nn
			batch_states = states.reshape([-1] + STATE_SHAPE)
			batch_advantages = advantages.reshape([-1])
			batch_returns = returns.reshape([-1])
			batch_actions = actions.reshape([-1])

				# estimated_return = rewards[t] + self.gamma * estimated_return * episodes_over_masks[t]
			feed_dict = {
				network.states: batch_states,
				network.action_indices: batch_actions,
				network.returns: batch_returns,
				network.advantages: batch_advantages
			}

			# gv = optimizer.compute_gradients(network.loss)
			# capped = [(tf.clip_by_global_norm(gv, 3.0), var) for grad, var in gv]
			# minimize_op = optimizer.apply_gradients(capped)

			ops = [minimize_op]
			_ = session.run(ops, feed_dict = feed_dict)

			log_header = "steps\tmean_r\tperf\taction counts"
			log_row = "{:.4f}\t{:.4f}\t{}"
			total_episode_reward += np.sum(rewards)
			action_counts = np.bincount(batch_actions.astype(np.uint8))
			mean_reward = np.mean(rewards)
			steps_per_second = (BATCH_SIZE*NUM_LEARNERS) / (time.time() - tic)
			log_msg = log_row.format(mean_reward, steps_per_second, action_counts)
			if i % 30 == 0:
				print(log_header)
			print(log_msg)

		print("total episode reward: {:.4f}".format(total_episode_reward))
		all_episode_rewards.append(total_episode_reward)

# ROM_PATH = r"../res/Mario Kart 64 (U) [!].z64"

# #special request
# def reward_func(env):
# 	return 0.0

# def run_instance(i, lock):
# 	env = MarioKartEnv(rom_path=ROM_PATH, instance_id=i, lock=lock, reward_func=reward_func)
# 	env.start()
# 	thread_name = threading.current_thread().name
# 	while True:
# 		reward = env.apply_action(3, frame_skip=4)
# 		if env.episode_step % 48 == 0:
# 			screen = env.get_current_state()
# 			filename = "env_{}_step_{}.png".format(i, env.episode_step)
# 			scaled = scipy.misc.imresize(screen, (128, 128))
# 			scipy.misc.imsave(filename, scaled)
# 		#print("[{}] position: {}, progress: {}, delta: {}".format(thread_name, env.position, env.lap_progress, env.progress_delta))

# if __name__ == "__main__":

# 	lock = multiprocessing.Event()

# 	num_instances = 4

# 	processes = []
# 	for i in range(0, num_instances):		
# 		p = multiprocessing.Process(target=run_instance, args=(i,lock,), daemon=True)
# 		processes.append(p)
# 		p.start()
# 		p.join(0.5)	# don't fully understand why this is necessary
# 		arrange_window(i)			

# 	# everything ready, emulation can begin
# 	lock.set()

# class SharedState():

# 	NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.uint8: c_uint}

# 	def __init__(self, environments):

# 		initial_states = np.array([env.get_current_state() for env in environments])
# 		self.screens = as_shared_array(initial_states)

# 		self.rewards = as_shared_array(np.zeros(num_instances, dtype=np.float32))
# 		self.is_done = as_shared_array(np.zeros(num_instances, dtype=np.uint8))

# 		action_counts = np.array([len(env.AVAILABLE_ACTIONS) for env in environments])
# 		self.actions = as_shared_array(action_counts, dtype=np.uint8)))

# 	def as_shared_array(self, array):
# 		dtype = self.NUMPY_TO_C_DTYPE[array.dtype.type]
# 		shape = array.shape
# 		shared = RawArray(dtype, array.reshape(-1))
#         return np.frombuffer(shared, dtype).reshape(shape)