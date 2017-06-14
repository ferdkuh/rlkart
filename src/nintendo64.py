from m64py.core.core import Core
from m64py.core.defs import *
import ctypes as C
from threading import Thread
import numpy as np
import time
from PIL import Image
from defs import *
from plugin_paths import *

class Nintendo64():
	
	def __init__(self):
		self.core = Core();
		self.allocate_buffer()
	
	def startup(self):
		self.core.core_load(dll_path)
		self.core.core_startup(dll_path, False)
		self.startup_plugins()		
		#self.register_frame_reader()

	def shutdown(self):
		#detach plugins
		self.core.stop()
		self.core.rom_close()
		self.core.core_shutdown()

	def run_game(self, rom_path):
		self.load_rom(rom_path)
		self.attach_plugins()
		self.thread = Thread(target=self.core.execute)
		self.thread.start()
		# how to detect when emu is ready for pause command?
		time.sleep(5)
		self.pause()

	def set_input_state(self, input_state):
		for i in range(0, len(INPUT_MAPPING)):
			if input_state(i):
				key_down(INPUT_MAPPING(i))
			else:
				key_up(INPUT_MAPPING(i))

	def key_down(self, key):
		self.core.send_sdl_keydown(key)

	def key_up(self, key):
		self.core.send_sdl_keyup(key)

	def get_frame(self):
		img = Image.frombytes('RGB', (200, 150), self.buffer, "raw")
		luminance = np.flipud((np.array(img) / 255.0).dot(RGB_TO_YUV))
		return luminance

	def pause(self):
		self.core.pause()

	def resume(self):
		self.core.resume()

	def step(self, num_frames=1):
		for i in range(0, num_frames):
			self.core.send_sdl_keydown(sdl2.SDLK_UP)
			self.core.advance_frame()
			# it seems you have to wait until the frame is ready?
			#time.sleep(0.1)

	def load_state(self, state_path):
		self.core.state_load(state_path)	

	# internal usage only
	def allocate_buffer(self):
		self.buffer = Framebuffer()
		buf_adr = C.addressof(self.buffer)
		self.buf_ptr = C.c_void_p(buf_adr)

	def register_frame_reader(self):
		self.callback = FRAME_CALLBACK_FUNC(self.read_screen_to_buffer)
		self.core.m64p.CoreDoCommand(M64CMD_SET_FRAME_CALLBACK, C.c_int(0), self.callback)

	def read_screen_to_buffer(self, frame_index):
		self.core.m64p.CoreDoCommand(M64CMD_READ_SCREEN, C.c_int(0), self.buf_ptr)		

	def startup_plugins(self):
		self.core.plugin_load_try(video_plugin_path)
		self.core.plugin_load_try(input_plugin_path)
		self.core.plugin_load_try(rsp_plugin_path)

		# startup plugins
		for plugin_type in PLUGIN_ORDER:
			# only one plugin of each type is loaded
			plugin_data = list(self.core.plugin_map[plugin_type].values())
			if plugin_data:
				handle, path, name, desc, version = plugin_data[0]
				self.core.plugin_startup(handle, name, desc)

	def attach_plugins(self):
		for plugin_type in PLUGIN_ORDER:
			plugin_data = list(self.core.plugin_map[plugin_type].values())
			if plugin_data:
				handle = plugin_data[0][0]
				self.core.m64p.CoreAttachPlugin(C.c_int(plugin_type), C.c_void_p(handle._handle))

	def load_rom(self, rom_path):
		# load rom
		with open(rom_path, "rb") as fs:
			romfile = fs.read(-1)
			rval = self.core.rom_open(romfile)
			if rval == M64ERR_SUCCESS:
				print("rom successfully loaded")
				self.core.rom_get_header()
				self.core.rom_get_settings()
			else:
				print("could not load rom.")

n64 = Nintendo64()
n64.startup()
#n64.run_game(r"k:\temp\mario64.z64")
n64.run_game(r"Mario Kart 64 (E) (V1.1) [!].z64")

def press(key, t=0.1):
	n64.key_down(key)
	time.sleep(t)
	n64.key_up(key)