import time
import multiprocessing
import threading
import sys
import numpy as np

from mariokart import MarioKartEnv



from PIL import Image
import scipy.misc
import tensorflow as tf

from myrl.learner_manager import LearnerProcessManager
from myrl.network import Network

ROM_PATH = r"../res/Mario Kart 64 (U) [!].z64"
def create_environment(index):
	return MarioKartEnv(ROM_PATH, index)

NUM_LEARNERS = 6
STATE_SHAPE = [84, 84]
BATCH_SIZE = 1

GAMMA = 0.99

def sample_action_from_policy(probabilities):
	# Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    # as seen in: https://github.com/Alfredvc/paac/blob/master/paac.py
	probabilities -= np.finfo(np.float32).epsneg
	action_indices = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probabilities]
	return action_indices

if __name__ == "__main__":

	manager = LearnerProcessManager(NUM_LEARNERS, create_environment)
	network = Network(num_actions = 9, state_shape = STATE_SHAPE)

	session = tf.InteractiveSession()
	optimizer = tf.train.RMSPropOptimizer(0.0224, decay=0.99, epsilon=0.1)
	minimize_op = optimizer.minimize(network.loss)

	# this blocks until each learner is ready
	session.run(tf.global_variables_initializer())
	manager.start_learners()
	# hack: do one no-op, so the learners will put something in their shared state array
	manager.update_learners([0]*NUM_LEARNERS)

	num_iterations = 500000

	states = np.zeros([BATCH_SIZE, NUM_LEARNERS] + STATE_SHAPE)
	rewards = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	values = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	returns = np.zeros([BATCH_SIZE, NUM_LEARNERS])
	actions = np.zeros([BATCH_SIZE, NUM_LEARNERS])

	for i in range(num_iterations):

		if i % 500 == 0:
			print("starting new episode")
			manager.start_new_episode()
			# temp = np.copy(manager.shared_memory.states)
			# for xx in range(NUM_LEARNERS):
			# 	frame = temp[xx]
			# 	img = Image.fromarray(frame)
			# 	filename = "learner_{}_step_{}.tiff".format(xx, i)
			# 	img.save(filename)
		print(i)

		current_state = np.copy(manager.shared_memory.states)
		for t in range(0, BATCH_SIZE):

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

			current_state = np.copy(manager.shared_memory.states)

		# # compute discounted returns and advantages for training
		estimated_return = value_out#np.zeros([NUM_LEARNERS])

		for t in reversed(range(BATCH_SIZE)):
			estimated_return = rewards[t] + GAMMA * estimated_return
			returns[t] = np.copy(estimated_return)

		# train nn
		batch_states = states.reshape([-1] + STATE_SHAPE)
		batch_values = values.reshape([-1])
		batch_returns = returns.reshape([-1])
		batch_actions = actions.reshape([-1])

		# 	estimated_return = rewards[t] + self.gamma * estimated_return * episodes_over_masks[t]
		feed_dict = {
			network.states: batch_states,
			network.action_indices: batch_actions,
			network.returns: batch_returns,
			network.values: batch_values
		}

		loss, _ = session.run([network.loss, minimize_op], feed_dict = feed_dict)
		
		if i == 0:
			print(batch_actions)
			
		log_format = "iteration: {}, loss: {:.4f}, avg reward: {:4f}, action counts: {}"
		#print(log_format.format(i, loss, np.mean(rewards), np.bincount(batch_actions.astype(np.uint8))))

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
