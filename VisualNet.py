import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import vgg


class VisualNet(nn.Module):

	def __init__(self, vgg16):
		super(VisualNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),            
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(256, 512, kernel_size=3, padding=1),            
			nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
		)

		self.conv5_pose = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)

		self.fc6_7 = nn.Sequential(
			nn.Linear(7 * 7 * 512, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.landmark_collar = nn.Linear(4096,10)
		self.landmark_sleeve = nn.Linear(4096,10)
		self.landmark_waistline = nn.Linear(4096,10)
		self.landmark_hem = nn.Linear(4096,10)
		
		self._initialize_weights()

	def forward(self, x):
		feature = self.features(x)
		l = self.conv5_pose(feature) 
		l = l.view(l.size(0), -1)
		l = self.fc6_7(l)
		collar = self.landmark_collar(l)
		sleeve = self.landmark_sleeve(l)
		waistline = self.landmark_waistline(l)
		hem = self.landmark_hem(l)

		return collar,sleeve,waistline,hem

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

def vsnet():
	model = VisualNet(vgg.vgg16(pretrained = True))

	return model

if __name__ == '__main__':
	vsnet()

