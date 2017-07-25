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

#self.conv5_pose = vgg.make_layers(['M',512,512,512,'M'])
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
		self.landmark_vis_loc_1 = nn.Linear(4096,6)
		self.landmark_vis_loc_2 = nn.Linear(4096,6)
		self.landmark_vis_loc_3 = nn.Linear(4096,6)
		self.landmark_vis_loc_4 = nn.Linear(4096,6)
		
		self._initialize_weights()
#self.landmark_visibility2 = nn.Linear(1024,2)
#		self.landmark_visibility3 = nn.Linear(1024,2)
#		self.landmark_visibility4 = nn.Linear(1024,2)

#		self.landmark_locationx1 = nn.Linear(1024,4)
#		self.landmark_locationx2 = nn.Linear(1024,4)
#		self.landmark_locationx3 = nn.Linear(1024,4)
#		self.landmark_locationx4 = nn.Linear(1024,4)

#self._init_weights()

	def forward(self, x):
		feature = self.features(x)
		l = self.conv5_pose(feature) 
		l = l.view(l.size(0), -1)
		l = self.fc6_7(l)
		l1 = self.landmark_vis_loc_1(l)
		l2 = self.landmark_vis_loc_2(l)
		l3 = self.landmark_vis_loc_3(l)
		l4 = self.landmark_vis_loc_4(l)

		return l1,l2,l3,l4

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

#lv1 = self.landmark_visibility1(l)
#		lv2 = self.landmark_visibility2(l)
#		lv3 = self.landmark_visibility3(l)
#		lv4 = self.landmark_visibility4(l)

#		ll1 = self.landmark_locationx1(l)
#		ll2 = self.landmark_locationx2(l)
#		l13 = self.landmark_locationx3(l)
#		l14 = self.landmark_locationx4(l)


def vsnet():
	model = VisualNet(vgg.vgg16(pretrained = True))

	return model

if __name__ == '__main__':
	vsnet()

