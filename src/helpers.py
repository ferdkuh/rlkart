import struct

def int_bits_to_float(value):
	return struct.unpack('f', struct.pack('I', value))[0]