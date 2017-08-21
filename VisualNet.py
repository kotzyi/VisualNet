import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import vgg


class VisualNet(nn.Module):

	def __init__(self):
		super(VisualNet, self).__init__()
		self.conv5_global = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		#self.pool5_local == nn.MaxPool2d(kernel_size = 2,stride = 2)
		self.fc6_local = nn.Sequential(
			nn.Linear(4 * 4 * 512 * 8, 1024),
			nn.ReLU(True),
			nn.Dropout(),
		)
		
		self.fc6_global = nn.Sequential(
			nn.Linear(7 * 7 * 512, 4096),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self.fc7_fusion = nn.Sequential(
			nn.Linear(1024 + 4096, 1024),
			nn.ReLU(True),
			nn.Dropout(),
		)
		self._initialize_weights()

	def forward(self, conv4, pool5):

		g = self.conv5_global(conv4)
		g = g.view(g.size(0), -1)
		g = self.fc6_global(g)

		#l = self.pool5_local(pool5)
		l = pool5.view(pool5.size(0), -1)
		l = self.fc6_local(l)
		
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

def visualnet():
	model = VisualNet()

	return model

if __name__ == '__main__':
	vsnet()

