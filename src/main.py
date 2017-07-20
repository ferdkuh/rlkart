# create timestep counter N and network
# create environments and learner agents

# main loop:

#	for t in range(0, max_episode_size):

#		get a_t[], v_t[] from network for the state of each agent:
#		convert a_t to a single action index
#		
#		parallel for i in range(0, num_agents)
#			new_state, reward = perform a_t[i] in environment[i]
#		
#		estimate R_tmax+1 for each agent
#		compute R_t for each agent
#		
#		train network

# needed
# network							# the neural network ops
# states							# array of states [N,84,84,4], shared memory
# shared_action_indices 			# array of int shape = [N]

# num_agents = 16
# max_episode_size = 30

# agent_manager = 0

# for t in range(0, max_episode_size):

# 	ops = [network.policy_out, network.value_out]
# 	feed_dict = { network.states: states }

# 	# policy_out has shape [num_agents, num_actions]
# 	# value out has shape [num_agents]
# 	policy_out, value_out = session.run(ops, feed_dict)
	
# 	# get one action index for each agent, write them to the shared memory
# 	shared_action_indices = sample_action_from_policy(policy_out)

# 	# run each environment for one timestep
# 	# blocks current until update is done
# 	agent_manager.update_agents()

# 	# copy results from shared array to episode buffer
import multiprocessing as mp
import numpy as np
import ctypes as C

import mariokart

import logging

logging.basicConfig(level=logging.DEBUG, format='(%(threadName)s) %(message)s',)

def sample_action_from_policy(probabilities):
	# Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    # as seen in: https://github.com/Alfredvc/paac/blob/master/paac.py
	probabilities -= np.finfo(np.float32).epsneg
	action_indices = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probabilities]
	return action_indices

# where can this function live?

	#return np.frombuffer(shared, dtype).reshape(shape)
	
NUM_ACTIONS = 8
ROM_PATH = r"../res/Mario Kart 64 (U) [!].z64"





