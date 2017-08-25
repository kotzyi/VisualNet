import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
	def __init__(self, visualnet):
		super(TripletNet, self).__init__()
		self.embeddingnet = visualnet #VisualNet model
		
	def forward(self, anchor_feature, anchor_pool5, positive_feature, positive_pool5, negative_feature, negative_pool5):
		""" anchor: Anchor feature map and pool5,
		negative: Distant (negative) feature map and pool5,
		positive: Close (positive) feature map and pool5 """
		embedded_anchor = self.embeddingnet(anchor_feature,anchor_pool5)
		embedded_positive = self.embeddingnet(positive_feature,positive_pool5)
		embedded_negative = self.embeddingnet(negative_feature,negative_pool5)
	
		dist_positive = F.pairwise_distance(embedded_anchor, embedded_positive, 2)
		dist_negative = F.pairwise_distance(embedded_anchor, embedded_negative, 2)
		
		return dist_positive, dist_negative

def tripletnet(embeddingnet):
	model = TripletNet(embeddingnet)
	return model

