import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import vgg


class VisualNet(nn.Module):

	def __init__(self, vgg16):
		super(VisualNet, self).__init__()
		self.features = nn.Sequential(*(vgg16.features[i] for i in range(23)))
		self.conv5_pose = vgg.make_layers(['M',512,512,512,'M'])
		self.fc6_7 = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1024),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.landmark_visibility1 = nn.Linear(1024,2)
		self.landmark_visibility2 = nn.Linear(1024,2)
		self.landmark_visibility3 = nn.Linear(1024,2)
		self.landmark_visibility4 = nn.Linear(1024,2)

		self.landmark_locationx1 = nn.Linear(1024,4)
		self.landmark_locationx2 = nn.Linear(1024,4)
		self.landmark_locationx3 = nn.Linear(1024,4)
		self.landmark_locationx4 = nn.Linear(1024,4)

#self._init_weights()

	def forward(self, x):
		feature = self.features(x)
		l = self.conv5_pose(feature) 
		l = l.view(l.size(0), -1)
		l = self.fc6_7(l)
		lv1 = self.landmark_visibility1(l)
		lv2 = self.landmark_visibility2(l)
		lv3 = self.landmark_visibility3(l)
		lv4 = self.landmark_visibility4(l)

		ll1 = self.landmark_locationx1(l)
		ll2 = self.landmark_locationx2(l)
		l13 = self.landmark_locationx3(l)
		l14 = self.landmark_locationx4(l)


def vsnet():
	model = VisualNet(vgg.vgg16(pretrained = True))

#	model = vgg.vgg16(pretrained = True)
#del model.classifier

#	model.vis = nn.Sequential(
#		nn.Linear(512 * 7 * 7, 4096),
#		nn.ReLU(True),
#		nn.Dropout(),
#		nn.Linear(4096, 1024),
#		nn.ReLU(True),
#		nn.Dropout(),
#		nn.Linear(1024, 24),
#	)

#	print(model.forward)
	for param in model.parameters():
		print(param.requires_grad)
	
	return model

if __name__ == '__main__':
	vsnet()

