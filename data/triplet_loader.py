from PIL import Image
from PIL import ImageFile
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

#ImageFile.LOAD_TRUNCATED_IMAGES = True
empty_value = -0.1

def default_loader(path):
	return Image.open(path).convert('RGB')

class TripletFolder(torch.utils.data.Dataset):
	def __init__(self, root, transform = None, target_transform = None, loader = default_loader):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader
		self.image_path = pd.read_table(root+'/Anno/list_landmarks_inshop2.txt',sep="\s+")['image_name'].values.tolist()
		self.triplets = pd.read_table(root+'/Anno/train_4.txt',sep="\s+")[['anchor', 'positive', 'negative']].values

		self.data_vis_loc = pd.read_table(root+'/Anno/list_landmarks_inshop2.txt',sep="\s+")[[
			'landmark_visibility_1','landmark_visibility_2','landmark_location_x_1','landmark_location_y_1','landmark_location_x_2','landmark_location_y_2',
			'landmark_visibility_3','landmark_visibility_4','landmark_location_x_3','landmark_location_y_3','landmark_location_x_4','landmark_location_y_4',
			'landmark_visibility_5','landmark_visibility_6','landmark_location_x_5','landmark_location_y_5','landmark_location_x_6','landmark_location_y_6',
			'landmark_visibility_7','landmark_visibility_8','landmark_location_x_7','landmark_location_y_7','landmark_location_x_8','landmark_location_y_8',
		]].values.tolist()
		self.image_size = [Image.open(self.root+path).size for path in self.image_path]
		
		# Normalization of coords
		for i in [2,4,8,10,14,16,20,22]:
			temp = (np.array(self.data_vis_loc, dtype=np.float32)[:,i] / np.array(self.image_size, dtype=np.float32)[:,0]).tolist()
			for j,x in enumerate(temp):
				if x > 0:
					self.data_vis_loc[j][i] = x
				else:
					self.data_vis_loc[j][i] = -0.1
		for i in [3,5,9,11,15,17,21,23]:
			temp = (np.array(self.data_vis_loc, dtype=np.float32)[:,i] / np.array(self.image_size, dtype=np.float32)[:,1]).tolist()
			for j,x in enumerate(temp):
				if x > 0:
					self.data_vis_loc[j][i] = x
				else:
					self.data_vis_loc[j][i] = -0.1

		for i in range(len(self.image_path)):
			for j in [0,1,6,7,12,13,18,19]:
				if self.data_vis_loc[i][j] == 1:
					self.data_vis_loc[i][j+j%2+2] = empty_value
					self.data_vis_loc[i][j+j%2+3] = empty_value

		self.data_vis_loc = np.array(self.data_vis_loc)
		
	def __getitem__(self, index):
#path = self.image_path[index]
		anchor, positive, negative = self.triplets[index]

		img1 = self.loader(self.root + self.image_path[int(anchor)])
		img2 = self.loader(self.root + self.image_path[int(positive)])
		img3 = self.loader(self.root + self.image_path[int(negative)])
		#################################### later this will be deleted #############################################
		target1 = self.data_vis_loc[int(anchor)]
		target2 = self.data_vis_loc[int(positive)]
		target3 = self.data_vis_loc[int(negative)]
		#############################################################
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
			img3 = self.transform(img3)
		return (img1, target1, img2, target2, img3, target3)

	def __len__(self):
		return len(self.triplets)
