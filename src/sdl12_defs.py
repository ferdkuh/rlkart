import ctypes as C
import numpy as np

# SDL 1.2 keycodes - damn them to hell!
SDL_KEY_UP 		= 273
SDL_KEY_DOWN 	= 274
SDL_KEY_RIGHT	= 275
SDL_KEY_LEFT	= 276
SDL_KEY_ENTER	= 13
SDL_KEY_LCTRL	= 306
SDL_KEY_LSHIFT	= 304
SDL_KEY_X		= 120
SDL_KEY_C 		= 99
SDL_KEY_Z		= 122
SDL_KEY_W 		= 119
SDL_KEY_A       = 97
SDL_KEY_S       = 115
SDL_KEY_D       = 100

# key to button mapping
ANALOG_LEFT 	= SDL_KEY_LEFT
ANALOG_RIGHT	= SDL_KEY_RIGHT
ANALOG_UP		= SDL_KEY_UP
ANALOG_DOWN		= SDL_KEY_DOWN
DPAD_LEFT		= SDL_KEY_A
DPAD_RIGHT		= SDL_KEY_D
DPAD_UP			= SDL_KEY_W
DPAD_DOWN		= SDL_KEY_S
TRIGGER_LEFT	= SDL_KEY_X
TRIGGER_RIGHT	= SDL_KEY_C
TRIGGER_Z		= SDL_KEY_Z
BUTTON_A		= SDL_KEY_LSHIFT
BUTTON_B		= SDL_KEY_LCTRL
BUTTON_START	= SDL_KEY_ENTER

ALL_KEYS = [ANALOG_LEFT, ANALOG_RIGHT, ANALOG_UP, ANALOG_DOWN,
			DPAD_LEFT, DPAD_RIGHT, DPAD_UP, DPAD_DOWN,
			TRIGGER_LEFT, TRIGGER_RIGHT, TRIGGER_Z,
			BUTTON_A, BUTTON_B, BUTTON_START]

# NO_OP			= 0
# ANALOG_LEFT 	= 1
# ANALOG_RIGHT 	= 1 << 2
# ANALOG_UP 		= 1 << 3
# ANALOG_DOWN 	= 1 << 4
# DPAD_LEFT 		= 1 << 5
# DPAD_RIGHT 		= 1 << 6
# DPAD_UP 		= 1 << 7
# DPAD_DOWN 		= 1 << 8
# TRIGGER_LEFT 	= 1 << 9
# TRIGGER_RIGHT   = 1 << 10
# TRIGGER_Z       = 1 << 11
# BUTTON_A 		= 1 << 12
# BUTTON_B 		= 1 << 13
# BUTTON_START 	= 1 << 14

# ALL_BUTTONS = [ANALOG_LEFT, ANALOG_RIGHT, ANALOG_UP, ANALOG_DOWN,
# 			   DPAD_LEFT, DPAD_RIGHT, DPAD_UP, DPAD_DOWN,
# 			   TRIGGER_LEFT, TRIGGER_RIGHT, TRIGGER_Z,
# 			   BUTTON_A, BUTTON_B, BUTTON_START]

# input vector layout, values should be 0 or 1
# this is actually mario kart specific, other game obviously need more buttons
# INPUT_MAPPING = [ANALOG_LEFT, ANALOG_RIGHT, BUTTON_A, BUTTON_B]

# types for interaction via ctypes
Framebuffer = C.c_ubyte*(200*150*3)
FRAME_CALLBACK_FUNC = C.CFUNCTYPE(None, C.c_uint)

# random shit
RGB_TO_Y = np.array([0.299, 0.587, 0.144])
