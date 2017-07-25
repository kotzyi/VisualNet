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
		self.landmark_vis_1 = nn.Linear(4096,2)
		self.landmark_vis_2 = nn.Linear(4096,2)
		self.landmark_vis_3 = nn.Linear(4096,2)
		self.landmark_vis_4 = nn.Linear(4096,2)
		self.landmark_vis_5 = nn.Linear(4096,2)
		self.landmark_vis_6 = nn.Linear(4096,2)
		self.landmark_vis_7 = nn.Linear(4096,2)
		self.landmark_vis_8 = nn.Linear(4096,2)

		self.landmark_loc_1 = nn.Linear(4096,4)
		self.landmark_loc_2 = nn.Linear(4096,4)
		self.landmark_loc_3 = nn.Linear(4096,4)
		self.landmark_loc_4 = nn.Linear(4096,4)
		
		self._initialize_weights()

	def forward(self, x):
		feature = self.features(x)
		l = self.conv5_pose(feature) 
		l = l.view(l.size(0), -1)
		l = self.fc6_7(l)
		v1 = self.landmark_vis_1(l)
		v2 = self.landmark_vis_2(l)
		v3 = self.landmark_vis_3(l)
		v4 = self.landmark_vis_4(l)
		v5 = self.landmark_vis_5(l)
		v6 = self.landmark_vis_6(l)
		v7 = self.landmark_vis_7(l)
		v8 = self.landmark_vis_8(l)

		l1 = self.landmark_loc_1(l)
		l2 = self.landmark_loc_2(l)
		l3 = self.landmark_loc_3(l)
		l4 = self.landmark_loc_4(l)
		return v1,v2,v3,v4,v5,v6,v7,v8,l1,l2,l3,l4

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

