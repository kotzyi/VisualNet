import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
	def __init__(self, embeddingnet):
		super(TripletNet, self).__init__()
		self.embeddingnet = embeddingnet #VisualNet model
		
	def forward(self, anchor, positive, negative):
		""" anchor: Anchor feature map and pool5,
		negative: Distant (negative) feature map and pool5,
		positive: Close (positive) feature map and pool5 """
		embedded_anchor = self.embeddingnet(anchor[0], anchor[1])
		embedded_positive = self.embeddingnet(positive[0], positive[1])
		embedded_negative = self.embeddingnet(negative[0], negative[1])
	
		dist_positive = F.pairwise_distance(embedded_anchor, embedded_positive, 2)
		dist_negative = F.pairwise_distance(embedded_anchor, embedded_negative, 2)
	
		return dist_a, dist_b
