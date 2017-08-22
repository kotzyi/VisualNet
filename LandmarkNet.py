import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import vgg


class LandmarkNet(nn.Module):
	def __init__(self, vgg16):
		super(LandmarkNet, self).__init__()
		self.conv5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)

		self.fc6 = nn.Sequential(
			nn.Linear(7 * 7 * 512, 4096),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.fc6_vis = nn.Sequential(
			nn.Linear(4096,1024),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.fc7 = nn.Sequential(
			nn.Linear(4096, 4072),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.fc8 = nn.Sequential(
			nn.Linear(4096,1024),
			nn.ReLU(True),
			nn.Dropout(),
		)

		self.landmark_vis_collar = nn.Linear(1024,6)
		self.landmark_vis_sleeve = nn.Linear(1024,6)
		self.landmark_vis_waistline = nn.Linear(1024,6)
		self.landmark_vis_hem = nn.Linear(1024,6)
		self.landmark_collar = nn.Linear(1024,4)
		self.landmark_sleeve = nn.Linear(1024,4)
		self.landmark_waistline = nn.Linear(1024,4)
		self.landmark_hem = nn.Linear(1024,4)
		
		self._initialize_weights()

		# For Transfer Learning
		self.features = nn.Sequential(
			*list(vgg16.features.children())[:23]
		)
		for param in self.features.parameters():
			param.requires_grad = False

	def forward(self, x):
		feature = self.features(x)
		l = self.conv5(feature) 
		l = l.view(l.size(0), -1)
		l = self.fc6(l)
		l_vis = self.fc6_vis(l)
		l1 = self.fc7(l)
		vis_collar = self.landmark_vis_collar(l_vis)
		vis_sleeve = self.landmark_vis_sleeve(l_vis)
		vis_waistline = self.landmark_vis_waistline(l_vis)
		vis_hem = self.landmark_vis_hem(l_vis)
		l = torch.cat((l1,vis_collar,vis_sleeve,vis_waistline,vis_hem),1)
		l = self.fc8(l)

		collar = self.landmark_collar(l)
		sleeve = self.landmark_sleeve(l)
		waistline = self.landmark_waistline(l)
		hem = self.landmark_hem(l)

		return vis_collar,vis_sleeve,vis_waistline,vis_hem,collar,sleeve,waistline,hem,feature

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

def landmarknet():
	model = LandmarkNet(vgg.vgg16(pretrained = True))
	return model

if __name__ == '__main__':
	vsnet()

