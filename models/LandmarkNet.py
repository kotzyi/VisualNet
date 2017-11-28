import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import math
import models.DRN as dilated_network

class LandmarkNet(nn.Module):
	def __init__(self, architect, types, resnet):
		super(LandmarkNet, self).__init__()
		self.types = types

		if architect in ['resnet18','resnet34','drn_22_d','drn_38_d','drn_54_d','drn_105_d']:
			in_ch = 512
			#in_ch = 1000
		else:
			in_ch = 2048

		out_ch = 4096 #4 #4096 #2048

#self.landmark_collar = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
#		self.landmark_sleeve = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
#		self.landmark_waistline = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
#		self.landmark_hem = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
		self.landmark_collar = nn.Sequential(
			nn.Linear(in_ch, out_ch),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(out_ch,4),
		)
		self.landmark_sleeve = nn.Sequential(
			nn.Linear(in_ch, out_ch),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(out_ch,4),
		)
		self.landmark_waistline = nn.Sequential(
			nn.Linear(in_ch, out_ch),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(out_ch,4),
		)
		self.landmark_hem = nn.Sequential(
			nn.Linear(in_ch, out_ch),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(out_ch,4),
		)
		self._initialize_weights()
		# For Transfer Learning
		self.features = nn.Sequential( # Must be changed when it's used for TripletTraining. 5->6 or 6->5 -> 4
			*list(resnet.children())[:5]
		)

		self.conv5 = nn.Sequential( #Must be changed when it's used for triplet training. 5:9 -> 6:9  when DRN 5:9 -> 5:10 -> 4:10
			*list(resnet.children())[5:9]
		)
		#for param in self.features.parameters():
		#			param.requires_grad = False

	def forward(self, x):
		feature = self.features(x)
		l = self.conv5(feature)
		l = l.view(l.size(0), -1)

		hem = self.landmark_hem(l)
		if self.types == 0:
			collar = self.landmark_collar(l)
			sleeve = self.landmark_sleeve(l)
			return collar,sleeve,hem,feature

		elif self.types == 1:
			waistline = self.landmark_waistline(l)
			return waistline,hem,feature

		else:
			collar = self.landmark_collar(l)
			sleeve = self.landmark_sleeve(l)
			waistline = self.landmark_waistline(l)
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

def landmarknet(architect, types):
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

