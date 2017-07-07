import struct
import time
import sys

if sys.platform == "win32":
	import win32gui

def int_bits_to_float(value):
	return struct.unpack('f', struct.pack('I', value))[0]

def arrange_window_win32(i, columns = 4, start_pos = (20, 20), spacing = 10):
	
	x0, y0 = start_pos

	hwnd = win32gui.FindWindow(None, "Mupen64Plus OpenGL Video Plugin by Rice v2.5.0")
	total_time_waited = 0.0
	while not hwnd:
		time.sleep(0.5)
		total_time_waited += 0.5
		hwnd = win32gui.FindWindow(None, "Mupen64Plus OpenGL Video Plugin by Rice v2.5.0")
		if total_time_waited > 10.0:
			print("[Warning] No matching window found after 5 seconds. Something is probably wrong!")
			total_time_waited = 0.0

	rect = win32gui.GetWindowRect(hwnd)
	rect = win32gui.GetWindowRect(hwnd)

	x, y = rect[0], rect[1]
	w, h = rect[2] - x, rect[3] - y	
	row = i // columns
	col = i % columns
	nx = x0 + col * w + (col-1) * spacing
	ny = y0 + row * h + (row-1) * spacing

	win32gui.SetWindowText(hwnd, "E{}".format(i))
	win32gui.MoveWindow(hwnd, nx, ny, w, h, True)

def arrange_window_x11(i):
	print("though luck. not implemented.")