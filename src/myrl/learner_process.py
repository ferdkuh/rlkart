import multiprocessing as mp
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG, format='(%(threadName)s) %(message)s',)

class LearnerProcess(mp.Process):

	def __init__(self, index, is_ready_barrier, queue, environment, shared_memory):
		super(LearnerProcess, self).__init__()
		self.is_ready_barrier = is_ready_barrier
		self.queue = queue
		self.environment = environment
		self.shared_memory = shared_memory
		self.index = index
		self.name = "Learner Process {}".format(index)

	def run(self):
		self.shared_memory.setup_np_wrappers()
		self.environment.start()
		#self.is_ready_barrier.wait()
		#self.shared_memory.states[self.index] = self.environment.get_current_state()
		while True:
			msg = self.queue.get()
			#logging.debug("[{}] received message: {}".format(self.index, msg))
			if msg == "new_episode":
				self.environment.new_episode()
				#hack: step one frame so the buffer updates?
				self.environment.step()
				self.shared_memory.states[self.index] = self.environment.get_current_state()
				self.is_ready_barrier.wait()				
			elif msg == "update":
				self.apply_next_action()
				# write new state and reward to shared memory
				new_state = self.environment.get_current_state()
				reward = self.environment.progress_delta
				self.shared_memory.rewards[self.index] = reward
				self.shared_memory.states[self.index] = new_state
				# signal that updated state has been written to shared memory
				self.is_ready_barrier.wait()
			elif msg == "stop":
				break

	def apply_next_action(self):
		action_index = self.shared_memory.action_indices[self.index]		
		self.environment.apply_action(action_index)
		