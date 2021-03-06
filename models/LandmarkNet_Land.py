import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import math
import models.DRN as dilated_network

class LandmarkNet(nn.Module):
	def __init__(self, architect, network):
		super(LandmarkNet, self).__init__()

		if architect in ['resnet18','resnet34','drn_26_c','drn_22_d','drn_38_d','drn_54_d','drn_105_d']:
			in_ch = 512
			hidden = 1024
		else:
			in_ch = 2048

		self.fc = nn.Sequential(
			nn.Conv2d(in_ch, hidden, kernel_size=1, stride=1, padding=0, bias=True),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Conv2d(hidden,hidden, kernel_size=1, stride=1, padding=0, bias=True),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Conv2d(hidden,hidden, kernel_size=1, stride=1, padding=0, bias=True),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Conv2d(hidden, 16, kernel_size=1, stride=1, padding=0, bias=True),
		)
		self.remain = nn.Sequential(
			nn.AvgPool2d (kernel_size=40, stride=40, padding=0, ceil_mode=False, count_include_pad=True),
		)
		

		#self._initialize_weights()

		# For Transfer Learning
		self.features = nn.Sequential( # Must be changed when it's used for TripletTraining. 5->6 or 6->5 -> 4
			*list(network.children())[:-2]
		)

	def forward(self, x):
		x  = self.features(x)
		x = self.remain(x)
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

def make_model(architect, pretrained = True):
	if architect == 'resnet50':
		model = LandmarkNet(architect, models.resnet50(pretrained=pretrained))
	elif architect == 'resnet101':
		model = LandmarkNet(architect, models.resnet101(pretrained=pretrained))
	elif architect == 'resnet152':
		model = LandmarkNet(architect, models.resnet152(pretrained=pretrained))
	elif architect == 'resnet34':
		model = LandmarkNet(architect, models.resnet34(pretrained=pretrained))
	elif architect == 'drn_26_c':
		model = LandmarkNet(architect, dilated_network.drn_c_26(pretrained=pretrained))
	elif architect == 'drn_22_d':
		model = LandmarkNet(architect, dilated_network.drn_d_22(pretrained=pretrained))
	elif architect == 'drn_38_d':
		model = LandmarkNet(architect, dilated_network.drn_d_38(pretrained=pretrained))
	elif architect == 'drn_54_d':
		model = LandmarkNet(architect, dilated_network.drn_d_54(pretrained=pretrained))
	elif architect == 'drn_105_d':
		model = LandmarkNet(architect, dilated_network.drn_d_105(pretrained=pretrained))
	else:
		model = LandmarkNet(architect, models.resnet18(pretrained=pretrained))

	return model

#if __name__ == '__main__':
#landmarknet()

