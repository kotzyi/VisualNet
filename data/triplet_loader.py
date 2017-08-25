from PIL import Image
from PIL import ImageFile
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

#ImageFile.LOAD_TRUNCATED_IMAGES = True

def default_loader(path):
	return Image.open(path).convert('RGB')

class TripletFolder(torch.utils.data.Dataset):
	def __init__(self, root, transform = None, target_transform = None, loader = default_loader):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader
		self.image_path = pd.read_table(root+'/Anno/list_landmarks_2.txt',sep="\s+")['image_name'].values.tolist()
		self.triplets = pd.read_table(root+'/Anno/triplet_val.txt',sep="\s+")[['anchor', 'positive', 'negative']].values
		
	def __getitem__(self, index):
		anchor, positive, negative = self.triplets[index]
		
		img1 = self.loader(self.root + self.image_path[int(anchor)-2])
		img2 = self.loader(self.root + self.image_path[int(positive)-2])
		img3 = self.loader(self.root + self.image_path[int(negative)-2])
	
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
			img3 = self.transform(img3)
			return (img1, img2, img3)
		else:
			return None

	def __len__(self):
		return len(self.triplets)
