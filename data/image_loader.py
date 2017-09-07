import torch.utils.data as data
from PIL import Image
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSION = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
	return Image.open(path).convert('RGB')

class AnchorFolder(data.Dataset):
	def __init__(self, root, transform = None, target_transform = None, loader = default_loader):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		img = self.loader(self.root)
		
		if self.transform is not None:
			img = self.transform(img)
		return img

	def __len__(self):
		return 1
