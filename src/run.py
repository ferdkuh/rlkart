import time
import multiprocessing
import sys

from mariokart import MarioKartEnv

if sys.platform == "win32":
	from helpers import arrange_window_win32 as arrange_window
else:
	from helpers import arrange_window_x11 as arrange_window

ROM_PATH = r"Mario Kart 64 (U) [!].z64"

#special request
def reward_func(env):
	return 0.0

def run_instance(i, lock):
	env = MarioKartEnv(rom_path=ROM_PATH, instance_id=i, lock=lock, reward_func=reward_func)
	env.start()
	while True:
		reward = env.apply_action(3, frame_skip=4)
		state = env.get_current_state()
		print("position: {}, progress: {}, delta: {}".format(env.position, env.lap_progress, env.progress_delta))

if __name__ == "__main__":

	lock = multiprocessing.Event()

	num_instances = 1

	processes = []
	for i in range(0, num_instances):		
		p = multiprocessing.Process(target=run_instance, args=(i,lock,), daemon=True)
		processes.append(p)
		p.start()
		p.join(0.5)	# don't fully understand why this is necessary
		arrange_window(i)			

	# everything ready, emulation can begin
	lock.set()