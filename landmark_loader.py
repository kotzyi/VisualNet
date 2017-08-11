import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import os
import json
import re
from glob import glob
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

image_size = 256

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSION = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
	return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
	def __init__(self, root, transform = None, target_transform = None, loader = default_loader):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

		self.data_frame = pd.read_table(root+"/Anno/list_landmarks.txt",sep="\s+")

		self.image_path = self.data_frame['image_name'].values.tolist()
		self.clothes_type = self.data_frame['clothes_type'].values.tolist()
		self.data_vis_loc = self.data_frame[[
							'landmark_visibility_1','landmark_visibility_2','landmark_location_x_1','landmark_location_y_1','landmark_location_x_2','landmark_location_y_2',
							'landmark_visibility_3','landmark_visibility_4','landmark_location_x_3','landmark_location_y_3','landmark_location_x_4','landmark_location_y_4',
							'landmark_visibility_5','landmark_visibility_6','landmark_location_x_5','landmark_location_y_5','landmark_location_x_6','landmark_location_y_6',
							'landmark_visibility_7','landmark_visibility_8','landmark_location_x_7','landmark_location_y_7','landmark_location_x_8','landmark_location_y_8',
		]].values

		self.image_size = [Image.open(self.root+path).size for path in self.image_path]
		
		# Normalization of coords
		for i in [2,4,8,10,14,16,20,22]:
			self.data_vis_loc[:,i] = (np.array(self.data_vis_loc)[:,i]/np.array(self.image_size)[:,0]).tolist()
			self.data_vis_loc[:,i] = [x if x > 0 else -1. for x in self.data_vis_loc[:,i]]

		for i in [3,5,9,11,15,17,21,23]:
			self.data_vis_loc[:,i] = (np.array(self.data_vis_loc)[:,i]/np.array(self.image_size)[:,1]).tolist()
			self.data_vis_loc[:,i] = [x if x > 0 else -1. for x in self.data_vis_loc[:,i]]

	def __getitem__(self, index):
		path = self.image_path[index]
		clothes_type = self.clothes_type[index]
		target = self.data_vis_loc[index]
		img = self.loader(self.root + path)
		#수정필요 왜냐하면 0 1 2 로 맞춰서 없는 것으로 변환시켜야 함.
		empty = np.array([2.,2.,-1.,-1.,-1.,-1.])
		mask = np.array([-1.,-1.])
		#vis = target[0:2] +target[6:8] + target[12:14] + target[18:20]
		if clothes_type == 1:
			collar = target[0:6]
			sleeve = target[6:12]
			hem = target[12:18]
			waistline = empty
		elif clothes_type == 2:
			waistline = target[0:6]
			hem = target[6:12]
			collar = empty
			sleeve = empty
		else:
			collar = target[0:6]
			sleeve = target[6:12]
			waistline = target[12:18]
			hem = target[18:24]

		if collar[0] == 1:
			collar[2:4] = mask
		if collar[1] == 1:
			collar[4:6] = mask

		if sleeve[0] == 1:
			sleeve[2:4] = mask
		if sleeve[1] == 1:
			sleeve[4:6] = mask

		if waistline[0] == 1:
			waistline[2:4] = mask
		if waistline[1] == 1:
			waistline[4:6] = mask

		if hem[0] == 1:
			hem[2:4] = mask
		if hem[1] == 1:
			hem[4:6] = mask

		if self.transform is not None:
			img = self.transform(img)
		return (img, clothes_type, collar, sleeve, waistline, hem, path)

	def __len__(self):
		return len(self.image_path)
