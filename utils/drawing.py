import cv2
import numpy as np

class Drawing():
	def __init__(self):
		self.colors = [(0,0,255),(255,0,0),(0,255,0),(255,255,0)]

		self.radius = 5
		self.counter = 0

	def draw_landmark_point(self, img_path, color, points):
		img = cv2.imread(img_path)

		for i, landmark in enumerate(zip(points[::2], points[1::2])):
			if landmark[0] > 0 and landmark[1] > 0:
				img = cv2.circle(img, (int(landmark[0]),int(landmark[1])), self.radius, color, -1)
			
			#if self.counter % 2 == 0: i += 1

		cv2.imwrite("img//"+str(self.counter)+".jpg", img)
		self.counter += 1

	def save_img(self, img, img_path):
		cv2.imwrite("new_" + img_path, img)
		print(img_path + " written")

