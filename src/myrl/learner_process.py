import multiprocessing as mp
import numpy as np
import PIL

# represents an environment that runs in a separate process
# it can be controlled via its message queue
class LearnerProcess(mp.Process):

	def __init__(self, index, is_ready_barrier, queue, environment, shared_memory):
		super(LearnerProcess, self).__init__()
		# signals that this process has finished some computation (e.g. apply action)
		self.is_ready_barrier = is_ready_barrier
		self.queue = queue
		self.environment = environment
		self.shared_memory = shared_memory
		self.index = index
		self.name = "Learner Process {}".format(index)
		self.daemon = True

	def run(self):
		# create the "numpy view" on the shared memory
		self.shared_memory.setup_np_wrappers()
		self.environment.start()

		# just here for recording movies, so ignore it in other actions
		frame_counter = 0

		# main message loop
		while True:
			msg = self.queue.get()
			if msg == "new_episode":
				self.environment.new_episode()
				self.environment.step()
				# copy the current state to the shared memory
				self.shared_memory.states[self.index] = self.environment.get_current_state()
				self.is_ready_barrier.wait()				
			elif msg == "update":
				self.apply_next_action()
				# get current state and reward and copy both to shared memory
				new_state = self.environment.get_current_state()
				reward = self.environment.progress_delta * 10.0
				self.shared_memory.rewards[self.index] = reward
				self.shared_memory.states[self.index] = new_state
				# signal that this process is done updating
				self.is_ready_barrier.wait()
			elif msg == "generate_frame":
				self.apply_next_action()
				if frame_counter % 4 == 0:
					frame = self.environment.n64.get_frame()
					frame = frame.transpose(PIL.Image.FLIP_TOP_BOTTOM)
					frame.save("frames/learner_{}_{:05d}.png".format(self.index, frame_counter))
				
				new_state = self.environment.get_current_state()
				reward = self.environment.progress_delta * 10.0
				self.shared_memory.rewards[self.index] = reward
				self.shared_memory.states[self.index] = new_state
									
				frame_counter += 1
				self.is_ready_barrier.wait()
			elif msg == "stop":
				break

	def apply_next_action(self, frame_skip=1):
		action_index = self.shared_memory.action_indices[self.index]		
		self.environment.apply_action(action_index, frame_skip)
		