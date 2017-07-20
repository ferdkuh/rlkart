# from mariokart import MarioKartEnv
# import threading

# ROM_PATH = r"../res/Mario Kart 64 (U) [!].z64"
# lock = threading.Event()

# def reward(env):
# 	return 0.0

# env = MarioKartEnv(rom_path=ROM_PATH, instance_id=0, lock=lock, reward_func=reward)
# env.start()

# def test():
# 	env.n64.core.state_load(r"../res/luigi_raceway_mario.state")
# 	env.n64.core.resume()
# 	env.n64.core.pause()
import numpy as np
import multiprocessing as mp
from datetime import datetime
import time
import random

from mariokart import MarioKartEnv

ROM_PATH = r"../res/Mario Kart 64 (U) [!].z64"

env = MarioKartEnv(ROM_PATH, 0)
env.start()

for i in range(500):
	env.apply_action(3)
	print(env.progress_delta)

# tic_time = datetime.now()

# def tic(msg=""):
# 	print(msg)
# 	tic_time = datetime.now()

# def toc():
# 	delta = datetime.now() - tic_time
# 	print(delta)

# num_learners = 2
# state_dims = np.array([2, 2])

# # tic("creating raw array")
# state_size = int(np.prod(state_dims))
# shared_array_size = num_learners * state_size

# temp = mp.RawArray(np.ctypeslib.ctypes.c_float, shared_array_size)
# # toc()

# # tic("assign to raw array")
# # temp[:] = np.arange(1e8, dtype = np.uint16)
# # toc()

# # tic("assign using memory view")
# #x = memoryview(temp).cast('B').cast('H')[:] #= np.arange(shared_array_size, dtype = np.float32)
# #x = memoryview(temp)[:] = np.arange(shared_array_size, dtype = np.float32)
# #print("memoryview len: ", len(x))
# # toc()
# class Learner():

# 	def __init__(self, index, shared_state_raw_array, shared_reward_raw_array):
# 		offset = state_size * index
# 		self.shared_state_array = np.frombuffer(shared_state_raw_array, np.float32)
# 		self.shared_reward_array = np.frombuffer(shared_reward_raw_array, np.float32)

# 	def get_stuff(self):
# 		new_state, reward = 3,3
# 		self.set_shared_state(new_state)
# 		self.set_shared_reward(reward)
# 		#tell some kind of locking mechanism that this thread is done

# 	def set_shared_state(self, state):
# 		offset = state_size * self.index
# 		self.shared_state_array[offset:offset+state_size] = state.reshape([-1])

# 	def set_shared_reward(self, reward):
# 		self.shared_reward_array[index] = reward

# import logging

# logging.basicConfig(level=logging.DEBUG, format='(%(processName)s) %(message)s',)

# def run(i, barrier, queue, condition):
# 	# n = 0
# 	#barrier.wait()
# 	while True:
# 		#logging.debug("waiting for actions in shared memory")
# 		barrier.wait()
# 		logging.debug("update environment, write (state,reward) to shared memory")
# 		#time.sleep(random.random() * 0.5)
		
# 		barrier.wait()		
# 		# n += 1
# 		#barrier.wait()

# def run_process(i, barrier, queue, condition):
# 	p = mp.Process(target=run, args=(i, barrier, queue, condition), name="runner {}".format(i))
# 	p.start()
# 	return p

# if __name__ == "__main__":

# 	num_learners = 8

# 	barrier = mp.Barrier(num_learners + 1)
# 	queues = []
# 	condition = mp.Condition()

# 	for i in range(0, num_learners):
# 		queue = mp.Queue()
# 		p = run_process(i, barrier, queue, condition)
# 		queues.append(queue)

	
# 	#barrier.reset()

# 	while True:
# 		#for q in queues: q.put(0)
# 		logging.debug("write action to shared memory")
# 		barrier.wait()
# 		#logging.debug("waiting for learner processes to write state reward to shared mem")
# 		barrier.wait()
# 		logging.debug("reading (s,r) from shared mem, doing computation")
# 		#time.sleep(random.random() * 0.5)

# 		#logging.debug("waiting for result")
# 		#barrier.wait()
# 		#logging.debug("waiting done, doing some processing")
# 		#time.sleep(1.0)
# 		#logging.debug("processing done, ready for more")

# # nparray = np.frombuffer(temp, np.float32)

# # # tic("create numpy wrapper")

# # # toc()

# # # tic("assign using numpy")
# # nparray[:] = np.arange(shared_array_size, dtype = np.float32)

# # s1 = nparray[0:state_size].reshape((2,2))
# # s2 = nparray[state_size:state_size*2].reshape((2,2))

# # print(s1)
# # print(s2)

# # print(list(temp))

# # s1[0,0] = 24.0

# # print(list(temp))