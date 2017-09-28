import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torchvision.models as models

class VisualNet(nn.Module):

	def __init__(self, resume, architect, types, resnet):
		super(VisualNet, self).__init__()
		
		self.types = types
		
		if architect in ['resnet18','resnet34']:
			local_in = 128
			in_ch = 512
		else:
			local_in = 512
			in_ch = 2048
		out_ch = 1024

		self.fc6_local = nn.Sequential(
			nn.Linear(4 * 4 * local_in * 8, 1024),
			nn.ReLU(True),
			nn.Dropout(),
		)
		
		self.fc6_global = nn.Sequential(
			nn.Linear(in_ch, out_ch),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.fc7_fusion = nn.Sequential(
#nn.Linear(1024 + out_ch, 1024),
			nn.Linear(4 * 4 * local_in * 8 + in_ch, 1024),
			nn.ReLU(True),
			nn.Dropout(),
		)
		if not resume:
			self._initialize_weights()

		self.conv5_global = nn.Sequential(
			*list(resnet.children())[6:9]
		)


	def forward(self, conv4, pool5):

		g = self.conv5_global(conv4)
		g = g.view(g.size(0), -1)
#g = self.fc6_global(g)

		l = pool5.view(pool5.size(0), -1)
#l = self.fc6_local(l)
		
		g = torch.cat((g,l),1)
		g = self.fc7_fusion(g)

		return g

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


def visualnet(resume, architect, types):
	if architect == 'resnet50':
		model = VisualNet(resume, architect, types, models.resnet50(pretrained=True))
	elif architect == 'resnet101':
		model = VisualNet(resume, architect, types, models.resnet101(pretrained=True))
	elif architect == 'resnet152':
		model = VisualNet(resume, architect, types, models.resnet152(pretrained=True))
	elif architect == 'resnet34':
		model = VisualNet(resume, architect, types, models.resnet34(pretrained=True))
	else:
		model = VisualNet(resume, architect, types, models.resnet18(pretrained=True))
	
	return model


if __name__ == '__main__':
	vsnet()

