import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import math
import models.DRN as dilated_network

class LandmarkNet(nn.Module):
	def __init__(self, architect, resnet):
		super(LandmarkNet, self).__init__()

		if architect in ['resnet18','resnet34','resnet50','resnet101','resnet152']:
			if architect in ['resnet18','resnet34']:
				in_ch = 512
			else:
				in_ch = 2048
			out_ch = 2048

			self.category = nn.Sequential(
				nn.Linear(in_ch, out_ch),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(out_ch, out_ch),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(out_ch,50),
			)
			self.attribute = nn.Sequential(
				nn.Linear(in_ch, out_ch),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(out_ch, out_ch),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(out_ch,1000),
			)

			#self._initialize_weights()
			"""
			self.features = nn.Sequential( # Must be changed when it's used for TripletTraining. 5->6 or 6->5 -> 4
				*list(resnet.children())[:7]
			)"""
			self.features = nn.Sequential(
				*list(resnet.children())[:5]
			)
			self.remain = nn.Sequential(
				*list(resnet.children())[5:9]
			)
			#self.remain= nn.AvgPool2d (kernel_size=10, stride=10, padding=0, ceil_mode=False, count_include_pad=True)
		else:
			in_ch = 512
			out_ch = 2048
			self.category = nn.Sequential(
				nn.Linear(in_ch, out_ch),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(out_ch, out_ch),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(out_ch,50),
			)
			self.attribute = nn.Sequential(
				nn.Linear(in_ch, out_ch),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(out_ch, out_ch),
				nn.ReLU(True),
				nn.Dropout(),
				nn.Linear(out_ch,1000),
			)
			#self._initialize_weights()

			self.features = nn.Sequential( # Must be changed when it's used for TripletTraining. 5->6 or 6->5 -> 4
				*list(resnet.children())[:11]
			)
			self.remain= nn.AvgPool2d (kernel_size=40, stride=40, padding=0, ceil_mode=False, count_include_pad=True)

	def forward(self, x):
		feature = self.features(x)
		x = self.remain(feature)
		x = x.view(x.size(0), -1)
		category = self.category(x)
		attribute = self.attribute(x)
		return feature, attribute, category


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

def make_model(architect, pretrained=True):
	if architect == 'resnet50':
		model = LandmarkNet(architect, models.resnet50(pretrained=pretrained))
	elif architect == 'resnet101':
		model = LandmarkNet(architect, models.resnet101(pretrained=pretrained))
	elif architect == 'resnet152':
		model = LandmarkNet(architect, models.resnet152(pretrained=pretrained))
	elif architect == 'resnet34':
		model = LandmarkNet(architect, models.resnet34(pretrained=pretrained))
	elif architect == 'drn_22_d':
		model = LandmarkNet(architect, dilated_network.drn_d_22(pretrained=pretrained))
	elif architect == 'drn_38_d':
		model = LandmarkNet(architect, dilated_network.drn_d_38(pretrained=pretrained))
	elif architect == 'drn_54_d':
		model = LandmarkNet(architect, dilated_network.drn_d_54(pretrained=pretrained))
	elif architect == 'drn_105_d':
		model = LandmarkNet(architect, dilated_network.drn_d_105(pretrained=pretrained))
	elif architect == 'drn_26_c':
		model = LandmarkNet(architect, dilated_network.drn_c_26(pretrained=pretrained))
	elif architect == 'drn_42_c':
		model = LandmarkNet(architect, dilated_network.drn_c_42(pretrained=pretrained))
	elif architect == 'drn_58_c':
		model = LandmarkNet(architect, dilated_network.drn_c_58(pretrained=pretrained))
	else:
		model = LandmarkNet(architect, models.resnet18(pretrained=pretrained))

	return model


