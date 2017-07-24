import multiprocessing as mp
import numpy as np
import sys

from myrl.shared_variables import SharedVariables
from myrl.learner_process import LearnerProcess

if sys.platform == "win32":
	from helpers import arrange_window_win32 as arrange_window
else:
	from helpers import arrange_window_x11 as arrange_window

class LearnerProcessManager():

	def __init__(self, num_learners, create_environment_func):
		self.create_environment_func = create_environment_func
		self.is_ready_barrier = mp.Barrier(num_learners+1)
		self.queues = [mp.Queue() for i in range(0, num_learners)]
		self.shared_memory = SharedVariables(num_learners, [84, 84])
		self.shared_memory.setup_np_wrappers()
		self.learners = [self.create_learner(i) for i in range(0, num_learners)]
		#self.start_learners()

	def stop_learners(self):
		for q in self.queues: q.put("stop")

	def start_learners(self):
		for learner in self.learners:
			learner.start()
			learner.join(0.5)
			arrange_window(learner.index, columns=3)
			# p = mp.Process(target=learner.run)
			# p.start()
			# p.join(0.5)
		#self.barrier.wait()

	def start_new_episode(self):
		for q in self.queues: q.put("new_episode")
		self.is_ready_barrier.wait()

	# this method updates all environments with the given action indices
	# and then blocks until all envs have written the new state and reward
	# to the shared memory
	def update_learners(self, action_indices):
		# write the actions to shared memory, then join barrier to signal that action has been written
		self.shared_memory.action_indices[:] = action_indices
		#self.barrier.wait()
		for q in self.queues: q.put("update")
		# wait until each learner has written new state and reward to shared memory
		self.is_ready_barrier.wait()
		# can read safely from shared memory
		# copy (state, reward) to episode buffer

	def create_learner(self, index):
		environment = self.create_environment_func(index)
		#mariokart.MarioKartEnv(ROM_PATH, index)
		learner = LearnerProcess(index, self.is_ready_barrier, self.queues[index], environment, self.shared_memory)
		return learner