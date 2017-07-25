import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import os
import json
import re
from glob import glob
import pandas
from pandas import Series, DataFrame

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
	def __init__(self, root, data_frame, transform = None, target_transform = None, loader = default_loader):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

		self.image_path = data_frame['image_name'].values
		self.clothes_type = data_frame['clothes_type'].values
		self.data_vis_loc = data_frame[[
							'landmark_visibility_1','landmark_visibility_2','landmark_visibility_3','landmark_visibility_4',
							'landmark_visibility_5','landmark_visibility_6','landmark_visibility_7','landmark_visibility_1',
							'landmark_location_x_1','landmark_location_y_1','landmark_location_x_2','landmark_location_y_2',
							'landmark_location_x_3','landmark_location_y_3','landmark_location_x_4','landmark_location_y_4',
							'landmark_location_x_5','landmark_location_y_5','landmark_location_x_6','landmark_location_y_6',
							'landmark_location_x_7','landmark_location_y_7','landmark_location_x_8','landmark_location_y_8',
		]].values

	def __getitem__(self, index):
		path = self.image_path[index]
		clothes_type = self.clothes_type[index]
		target = self.data_vis_loc[index]
		img = self.loader(self.root + path)
		if self.transform is not None:
			img = self.transform(img)
		return (img, ,clothes_type, target)

	def __len__(self):
		return len(self.data_path)
