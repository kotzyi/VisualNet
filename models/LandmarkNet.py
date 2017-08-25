import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import math

class LandmarkNet(nn.Module):
	def __init__(self, resnet):
		super(LandmarkNet, self).__init__()
		self.fc6 = nn.Sequential(
			nn.Linear(2048, 4096),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.landmark_collar = nn.Sequential(
			nn.Linear(2048, 2048),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(2048,4),
		)
		self.landmark_sleeve = nn.Sequential(
			nn.Linear(2048, 2048),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(2048,4),
		)
		self.landmark_waistline = nn.Sequential(
			nn.Linear(2048, 2048),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(2048,4),
		)
		self.landmark_hem = nn.Sequential(
			nn.Linear(2048, 2048),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(2048,4),
		)
		
		self._initialize_weights()

		# For Transfer Learning
		self.features = nn.Sequential(
			*list(resnet.children())[:6]
		)
		self.conv5 = nn.Sequential(
			*list(resnet.children())[6:9]
		)
		#for param in self.features.parameters():
		#			param.requires_grad = False

	def forward(self, x):
		feature = self.features(x)
		l = self.conv5(feature) 
		l = l.view(l.size(0), -1)

		collar = self.landmark_collar(l)
		sleeve = self.landmark_sleeve(l)
		waistline = self.landmark_waistline(l)
		hem = self.landmark_hem(l)
		
		return collar,sleeve,waistline,hem,feature

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

def landmarknet(architect):
	if architect == 'resnet50':
		model = LandmarkNet(models.resnet50(pretrained=True))
	elif architect == 'resnet101':
		model = LandmarkNet(models.resnet101(pretrained=True))
	elif architect == 'resnet152':
		model = LandmarkNet(models.resnet152(pretrained=True))

	return model

if __name__ == '__main__':
	vsnet()

