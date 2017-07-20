import multiprocessing as mp
import numpy as np
import ctypes as C

NUMPY_TO_C_DTYPE = {
	np.float32: C.c_float, 
	np.float64: C.c_double, 
	np.uint8: C.c_byte
}

def as_shared_array(array):
	dtype = NUMPY_TO_C_DTYPE[array.dtype.type]
	shape = array.shape
	raw_array = mp.RawArray(dtype, array.reshape(-1))
	return raw_array

# just nice named references to the shared arrays
class SharedVariables():

	def __init__(self, num_learners, state_shape):
		self.num_learners = num_learners
		self.state_shape = state_shape
		self.raw_action_indices = as_shared_array(np.zeros(num_learners, dtype=np.uint8))
		self.raw_rewards = as_shared_array(np.zeros(num_learners, dtype=np.float32))
		self.raw_states = as_shared_array(np.zeros([num_learners] + state_shape, dtype=np.float32))

	def setup_np_wrappers(self):
		self.action_indices = np.frombuffer(self.raw_action_indices, dtype=np.uint8).reshape(self.num_learners)
		self.rewards = np.frombuffer(self.raw_rewards, dtype=np.float32).reshape(self.num_learners)
		self.states = np.frombuffer(self.raw_states, dtype=np.float32).reshape([self.num_learners] + self.state_shape)
