import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import os
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

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
	def __init__(self, root, is_train,transform = None, target_transform = None, loader = default_loader):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

		if is_train:
			self.data_frame = pd.read_table(root+"/Anno/list_landmarks_train.txt",sep="\s+")
		else:
			self.data_frame = pd.read_table(root+"/Anno/list_landmarks_test.txt",sep="\s+")

		self.image_path = self.data_frame['image_name'].values.tolist()
		self.clothes_type = self.data_frame['clothes_type'].values.tolist()

	def __getitem__(self, index):
		path = self.image_path[index]
		clothes_type = self.clothes_type[index] - 1
		img = self.loader(self.root + path)

		if self.transform is not None:
			img = self.transform(img)
		return (img, clothes_type)

	def __len__(self):
		return len(self.image_path)
