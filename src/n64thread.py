import sys
import threading
import helpers
import platform

from threading import Thread, Event
from m64py.core.core import Core
from m64py.core.defs import *
from defs import *
from PIL import Image

if sys.platform == "win32":
    from _ctypes import FreeLibrary as dlclose    
else:
    from _ctypes import dlclose    

if platform.architecture()[0] == "32bit":
	print("using 32bit mupen64plus lib")
	from plugin_paths_32 import *
else:
	print("using 64bit mupen64plus lib")
	from plugin_paths import *

class N64GameThread(Thread):

	#thread init and emulator setup
	def __init__(self, rom_path, instance_id=-1):
		super(N64GameThread, self).__init__()
		self.name = "N64Thread {}".format(instance_id)
		self.daemon = True
		self.rom_path = rom_path
		self.core = Core()

		# create the buffer that will hold the current screen
		self.framebuffer = Framebuffer()
		buf_adr = C.addressof(self.framebuffer)
		self.buf_ptr = C.c_void_p(buf_adr)

		self.init()

	def init(self):
		self.core.core_load(dll_path)
		if self.core.get_handle():
			self.core.core_startup(dll_path, False)
			self.load_plugins()
			self.apply_plugin_action(self.startup_plugin)

			# self.flag = Event()
			# self.flag.clear()
			self.frame_callback_func = FRAME_CALLBACK_FUNC(self.frame_callback)	
			self.core.m64p.CoreDoCommand(M64CMD_SET_FRAME_CALLBACK, None, self.frame_callback_func)
		else:
			print("[Error] could not load mupen64plus core library")

	def stop(self):
		self.core.stop()
		self.apply_plugin_action(self.shutdown_plugin)
		self.apply_plugin_action(self.unload_plugin)
		self.core.core_shutdown()
		self.unload_library(self.core.get_handle())
		self.core.m64p = None
		self.core.config = None

	def run(self):
		self.load_rom()
		self.apply_plugin_action(self.attach_plugin)
		self.core.execute()
		#self.core.detach_plugins()
		self.apply_plugin_action(self.detach_plugin)
		#self.detach_plugins()
		self.close_rom()
		self.stop()

	# user methods
	def set_input_state(self, input_list):
		for key in ALL_KEYS:
			if key in input_list:
				self.core.send_sdl_keydown(key)
			else:
				self.core.send_sdl_keyup(key)

	def get_frame(self):
		img = Image.frombytes('RGB', (200, 150), self.framebuffer, "raw")
		return img

	def read_memory_float32(self, address):
		bits = self.core.m64p.DebugMemRead32(address)
		return helpers.int_bits_to_float(bits)

	def read_memory_uint8(self, address):
		return self.core.m64p.DebugMemRead8(address)

	def read_memory_uint16(self, address):
		return self.core.m64p.DebugMemRead16(address)

	def read_memory_uint32(self, address):
		return self.core.m64p.DebugMemRead32(address)

	# more convenient methods
	def get_video_size(self):
		rval = self.core.core_state_query(M64CORE_VIDEO_SIZE)
		return self.decode_video_size(rval)

	#temp debug
	
		# print("i'm a callback, i'm a callback")
		# print("frame done!")
		# thread_name = threading.current_thread().name
		# print("[{}] -> frame_callback".format(thread_name))
		# self.flag.set()

	# def set_video_size(self, width, height):
	# 	video_size = (width << 16) + height
	# 	rval = self.core.core_state_set(M64CORE_VIDEO_SIZE, video_size)
		# return rval

	# def step(self):
	# 	thread_name = threading.current_thread().name
	# 	print("[{}] -> step".format(thread_name))
	# 	self.flag.clear()
	# 	self.core.advance_frame()
	#end temp

	# private methods, nothing to see here
	def frame_callback(self, frame_index):
		self.core.m64p.CoreDoCommand(M64CMD_READ_SCREEN, C.c_int(0), self.buf_ptr)

	def decode_video_size(self, video_size):
		x = video_size >> 16
		y = video_size & 0x0000ffff
		return x,y

	def load_plugins(self):
		self.core.plugin_load_try(video_plugin_path)
		self.core.plugin_load_try(input_plugin_path)
		self.core.plugin_load_try(rsp_plugin_path)

	def unload_plugin(self, handle, name, plugin_type, desc):
		self.unload_library(handle)

	def startup_plugin(self, handle, name, plugin_type, desc):
		self.core.plugin_startup(handle, name, desc)

	def shutdown_plugin(self, handle, name, plugin_type, desc):
		self.core.plugin_shutdown(handle, desc)

	def attach_plugin(self, handle, name, plugin_type, desc):
		self.core.m64p.CoreAttachPlugin(C.c_int(plugin_type), C.c_void_p(handle._handle))

	def detach_plugin(self, handle, name, plugin_type, desc):	
		self.core.m64p.CoreDetachPlugin(plugin_type)

	def detach_plugins(self):		
		for plugin_type in PLUGIN_ORDER:		
			rval = self.core.m64p.CoreDetachPlugin(plugin_type)		
			print("detached plugin '{}': {}".format(plugin_type, rval))

	def apply_plugin_action(self, func):
		# only one plugin of each type is loaded, so this should be fine
		for plugin_type in PLUGIN_ORDER:					
			plugin_data = list(self.core.plugin_map[plugin_type].values())
			if plugin_data:
				handle, path, name, desc, version = plugin_data[0]
				func(handle, name, plugin_type, desc)

	def unload_library(self, handle):
		dlclose(handle._handle)
		del handle

	def load_rom(self):
		with open(self.rom_path, "rb") as fs:
			romfile = fs.read(-1)
			rval = self.core.rom_open(romfile)
			if rval == M64ERR_SUCCESS:
				del romfile
				self.core.rom_get_header()
				self.core.rom_get_settings()
			else:
				print("[ERROR] could not load rom.")

	def close_rom(self):
		self.core.rom_close()