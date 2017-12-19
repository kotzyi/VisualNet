import torch
from torchvision import transforms

class Data_Manage():
	def __init__(self, batch_size, IMG_SIZE = 256, workers = 16, CUDA = True):
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.trans = transforms.Compose([
			transforms.Scale(IMG_SIZE),
			transforms.CenterCrop(IMG_SIZE),
			transforms.ToTensor(),
			self.normalize,
		])
		self.CUDA = CUDA
		self.workers = workers
		self.batch_size = batch_size

	def load(self, module_name,shuffle,**kwargs):
		print("Data loading")
		folder = self.__load_module('data', module_name.replace('.py',''))

		d = torch.utils.data.DataLoader(
			folder.ImageFolder(kwargs,transform = self.trans),
			batch_size = self.batch_size,
			shuffle = shuffle,
			num_workers = self.workers,
			pin_memory = self.CUDA,
		)
		print("Complete Data loading ({})".format(len(d)))
		return d

	def __load_module(self, pkg_name, module_name):
		module = __import__("{0}.{1}".format(pkg_name, module_name), globals(), locals(), [module_name])
		return module
