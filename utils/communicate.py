import socket
import struct
import pickle

BUFFER_SIZE = 4096

class Communicate():
	def __init__(self, TCP_IP, TCP_PORT):
		self.TCP_IP = TCP_IP
		self.TCP_PORT = TCP_PORT
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.clients = {}
		self.count = 1
		self.payload_size = struct.calcsize("I")

	def receive(self, addr):
		try: 
			if self.sock.connect_ex(addr):
				conn = self.clients[addr]
		except:
			conn, addr = self.sock.accept()
			self.clients[addr] = conn

		print("Got connection from",addr)
		print("Receiving... {}".format(self.count))

		data = b''

		while len(data) < self.payload_size:
			data += conn.recv(BUFFER_SIZE)
		packed_msg_size = data[:self.payload_size]
		data = data[self.payload_size:]
		msg_size = struct.unpack("I", packed_msg_size)[0]
		
		while len(data) < msg_size:
			data += conn.recv(BUFFER_SIZE)
		frame_data = data[:msg_size]
		data = data[msg_size:]
		frame = pickle.loads(frame_data)

		self.count+=1

		return frame, addr

	def send(self, addr, data):
		self.clients[addr].send(data)
		print("Send data to '{}'".format(addr))

	def listen(self):
		self.sock.bind((self.TCP_IP, self.TCP_PORT))
		self.sock.listen(10)
		print("Listening...")

