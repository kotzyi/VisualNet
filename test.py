####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
### KT AI Commerce Team: Junyeop Lee.                                                                                            ###
### Code Written in 12.12.17                                                                                                     ###
### Description: This program is to purpose to test landmark detection of cloth by cam camera.                                   ###
###              TCP/IP is the protocol of this program between server and clients.                                              ###
###              The program is bascially loaded three different network models that are landmark model, visuality model.        ###
###              Visuality model inferences which there is a landmark coord on image. Landmark Model inferences where the        ###
###              landmark coordination on image. Another model can surely added to this code that you want.                      ###
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


from __future__ import print_function
import argparse
import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

import utils.model_manage as MM
import utils.communicate as comm
import utils.data_manage as DM
import utils.utility as utility

TCP_IP = '10.214.35.36'
TCP_PORT = 5005
BUFFER_SIZE = 4096
IMG_SIZE = 320
WINDOW_SIZE = 3

# Training settings
parser = argparse.ArgumentParser(description='Visual Search')
parser.add_argument('-data', metavar = 'DIR',help = 'path to dataset')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('-l','--land', nargs='+',default = [],
                    help='model file name, architecture, and path to latest checkpoint (ex: -l LandmarkNet_Land.py drn_38_d save/landmark/checkpoint.pth.tar)')
parser.add_argument('-v','--visuality', nargs='+',default = [],
					help='model file name, architecture, and path to latest checkpoint (ex: -v Landmakr_Vis.py resnet50 save/visuality/checkpoint.pth.tar)')
parser.add_argument('-a','--attribute', nargs='+',default = [],
					help='model file name, architecture, and path to latest checkpoint (ex: -a resnet.py resnet34 save/attribute/checkpoint.pth.tar)')
parser.add_argument('-f','--feature', nargs='+',default = [],
					help='model file name, architecture, and path to latest checkpoint (ex: -f resnet.py resnet34 save/attribute/checkpoint.pth.tar)')
parser.add_argument('-c','--category', nargs='+',default = [],
					help='model file name, architecture, and path to latest checkpoint (ex: -c resnet.py resnet34 save/attribute/checkpoint.pth.tar)')


trans = transforms.Compose([
	transforms.Scale(IMG_SIZE),
	transforms.CenterCrop(IMG_SIZE),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
	#기본 설정 부분
	global args, best_acc
	args = parser.parse_args()

	landmark_params = args.land
	visuality_params = args.visuality
	attribute_params = args.attribute
	category_params = args.category

	cuda = args.cuda and torch.cuda.is_available()

	if args.cuda:
		print("GPU Mode")
	else:
		print("CPU Mode")

	models = MM.Model_Manage()
	#Landmark Models Load
	if args.land:
		land_model = models.add(landmark_params[0:2])
		land_model, _ = models.chkpoint_load(land_model, ['epoch'], landmark_params[-1])
		if land_model is not None:
			land_model.eval()

	#Visuality Models Load
	if args.visuality:
		visual_model = models.add(visuality_params[0:2])
		visual_model, _ = models.chkpoint_load(visual_model, ['epoch'], visuality_params[-1])
		if visual_model is not None:
			visual_model.eval()

	#Attribute Models Load
	if args.attribute:
		attr_model = models.add(attribute_params[0:2])
		attr_model, _ = models.chkpoint_load(attr_model, ['epoch'], attribute_params[-1])
		if attr_model is not None:
			attr_model.eval()

	#Feature Models Load
	if args.feature:
		feature_model = models.add(feature_params[0:2])
		feature_model, _ = models.chkpoint_load(feature_model, ['epoch'], feature_params[-1])
		if feature_model is not None:
			feature_model.eval()

	#Category Models Load
	if args.category:
		category_model = models.add(category_params[0:2])
		category_model, _ = models.chkpoint_load(category_model, ['epoch'], category_params[-1])
		cat_class = utility.cat_class
		attr_class = utility.attr_class
		if category_model is not None:
			category_model.eval()


	#Connect with local for feature detection
	addr = ('0.0.0.0',0)
	connection = comm.Communicate(TCP_IP,TCP_PORT)
	connection.listen()

	while True:
		frame, addr = connection.receive(addr)
		input = Image.fromarray(frame)
		input_tensor = torch.unsqueeze(trans(input),0)
		input_var = Variable(input_tensor,volatile =True).cuda()

		landmark  = land_model(input_var).data.tolist()[0]
		#visuality = visual_model(input_var).data.tolist()[0]
		_,attribute,category = category_model(input_var)
		_,attr = torch.topk(attribute.data,3,1)
		_,cat = torch.topk(category.data,5,1)
		attr = attr[0,:]
		cat = cat[0,:]
		
		print("Attr:",attr_class[attr[0]],attr_class[attr[1]],attr_class[attr[2]])
		print("Category:",cat_class[cat[0]],cat_class[cat[1]],cat_class[cat[2]],cat_class[cat[3]],cat_class[cat[4]])

		coords = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		#coords = set_coords(landmark,visuality)
		coords=pickle.dumps(landmark)
		connection.send(addr, coords)

	return


if __name__ == '__main__':
	main()    
