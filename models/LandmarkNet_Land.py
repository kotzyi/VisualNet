import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import math
import models.DRN as dilated_network

class LandmarkNet(nn.Module):
	def __init__(self, architect, types, network):
		super(LandmarkNet, self).__init__()
		self.types = types

		if architect in ['resnet18','resnet34','drn_22_d','drn_38_d','drn_54_d','drn_105_d']:
#in_ch = 512
			in_ch = 512
		else:
			in_ch = 2048

		# For Transfer Learning
		self.features = nn.Sequential( # Must be changed when it's used for TripletTraining. 5->6 or 6->5 -> 4
			*list(network.children())[:-1]
		)
		self.fc = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Conv2d(in_ch, 16, kernel_size=1, stride=1, padding=0, bias=True),
		)

		#for param in self.features.parameters():
		#			param.requires_grad = False
	def forward(self, x):
		x = self.features(x)
		x = self.fc(x)
		x = x.view(x.size(0), -1)

		return x

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

def landmarknet_land(architect, types):
	if architect == 'resnet50':
		model = LandmarkNet(architect, types, models.resnet50(pretrained=True))
	elif architect == 'resnet101':
		model = LandmarkNet(architect, types, models.resnet101(pretrained=True))
	elif architect == 'resnet152':
		model = LandmarkNet(architect, types, models.resnet152(pretrained=True))
	elif architect == 'resnet34':
		model = LandmarkNet(architect, types, models.resnet34(pretrained=True))
	elif architect == 'drn_22_d':
		model = LandmarkNet(architect, types, dilated_network.drn_d_22(pretrained=True))
	elif architect == 'drn_38_d':
		model = LandmarkNet(architect, types, dilated_network.drn_d_38(pretrained=True))
	elif architect == 'drn_54_d':
		model = LandmarkNet(architect, types, dilated_network.drn_d_54(pretrained=True))
	elif architect == 'drn_105_d':
		model = LandmarkNet(architect, types, dilated_network.drn_d_105(pretrained=True))
	else:
		model = LandmarkNet(architect, types, models.resnet18(pretrained=True))

	return model

#if __name__ == '__main__':
#landmarknet()

