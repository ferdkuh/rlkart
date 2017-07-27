import time
import threading
import m64py.core.core as mp64core

from n64thread import N64GameThread
from m64py.core.defs import *
from defs import *

# this class represents an emulator instance running mariokart 64
class MarioKartEnv():

	AVAILABLE_ACTIONS = [
		[],							#0					
		[ANALOG_LEFT],				#1
		[ANALOG_RIGHT],				#2
		[BUTTON_A]#,					#3
		# [BUTTON_B],					#4
		# [ANALOG_LEFT, BUTTON_A],	#5
		# [ANALOG_LEFT, BUTTON_B],	#6
		# [ANALOG_RIGHT, BUTTON_A],	#7
		# [ANALOG_RIGHT, BUTTON_B]	#8
	]

	def __init__(self, rom_path, instance_id):
		self.instance_id = instance_id
		self.rom_path = rom_path
		#self.n64 = N64GameThread(rom_path, instance_id=instance_id)
		#self.lock = lock		
		#self.reward_func = reward_func

	def start(self):
		#t.core.state_load(r"luigi_raceway_mario.state")
		self.n64 = N64GameThread(self.rom_path, instance_id=self.instance_id)		
		self.init()
		#print("init done")
		#self.lock.wait()
		self.new_episode()	

	def init(self):
		threading.current_thread().name = "Worker_{}".format(self.instance_id)		
		self.n64.start()
		state = self.n64.core.core_state_query(M64CORE_EMU_STATE)
		while state != M64EMU_RUNNING:
			time.sleep(0.1)		
			state = self.n64.core.core_state_query(M64CORE_EMU_STATE)	
		time.sleep(1)	# scary, without this wait nothing works
		self.n64.core.pause()	

	# user methods
	def new_episode(self):		
		self.lap = 0
		self.position = 8
		self.lap_progress = 0.0
		self.progress_delta = 0.0
		self.last_lap_progress = 0.0
		self.race_started = False
		self.race_finished = False
		self.magic = False
		self.episode_step = 0
		self.n64.core.state_load(r"../res/luigi_raceway_mario.state")	

	def apply_action(self, action_index, frame_skip=1):
		input_list = MarioKartEnv.AVAILABLE_ACTIONS[action_index]
		self.n64.set_input_state(input_list)
		self.step(frame_skip)
		self.update_status_variables()

	def get_current_state(self):
		img = self.n64.get_frame()
		# scale down?
		img = img.resize((84, 84))
		luminance = np.flipud((np.array(img) / 255.0).dot(RGB_TO_Y))
		return luminance

	# helper methods for reward generation
	def raw_position(self):
		# [1,8]
		# this seems fishy, why does it return correct things?
		return self.n64.read_memory_uint32(0x801643B8) + 1

	def raw_lap(self):
		#[1,3]
		# if lap_progress > 0.9 and was never smaller, then lap = 0
		return self.n64.read_memory_uint8(0x8018CAE1) + 1 	# #101ACAE2, 101ACAE1

	def raw_lap_progress(self):
		# [0,1], but is ~0.99 at beginning because kart is in front of finish line
		return self.n64.read_memory_float32(0x801644A8)

	# private methods
	def step(self, n=1):
		for i in range(0,n):
			self.n64.core.advance_frame()
			mp64core.is_paused.wait()			# this needs to be made less ugly
			self.episode_step += 1		
		
	def update_status_variables(self):
		
		self.lap = self.raw_lap()
		self.position = self.raw_position()

		# overall progress in the current lap
		if not self.magic and self.raw_lap_progress() < 0.1:
			self.magic = True

		self.lap_progress = self.raw_lap_progress()
		if self.magic and self.lap == 1 and self.raw_lap_progress() > 0.9:
			self.lap_progress = 0.0

		# progress made on the track since last update
		current_progress = self.raw_lap_progress()
		delta = current_progress - self.last_lap_progress 
		if abs(delta) > 0.9: 
			delta = max(0.0, delta - 1.0)
		self.progress_delta = delta
		self.last_lap_progress = current_progress

		# has the race started yet
		if not self.race_started and self.raw_lap_progress() > 0.0:
			self.race_started = True

		# is the race finished
		self.race_finished = False
	