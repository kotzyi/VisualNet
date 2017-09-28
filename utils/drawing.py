import cv2
import numpy as np

class Drawing():
	def __init__(self):
		self.dot_color = (0,0,255)
		self.radius = 5

	def draw_dot(self, img_path, x, y):
		img = cv2.imread(img_path)
		img = cv2.circle(img, (x,y), self.radius, self.dot_color, -1)
		cv2.imwrite("new_" + img_path+".jpg", img)

	def save_img(self, img, img_path):
		cv2.imwrite("new_" + img_path, img)
		print(img_path + " written")

