####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
### train.py                                                                                                                     ###
### KT AI Commerce Team: Junyeop Lee.                                                                                            ###
### Code Written in 12.12.17                                                                                                     ###
### Description: This program is to purpose to train landmark detection of cloth.                                                ###
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import utils.model_manage as MM
import utils.communicate as comm
import utils.data_manage as DM
import utils.utility as utility

import landmark
import attribute

import time

IMG_SIZE = 224
WINDOW_SIZE = 3

# Training settings
parser = argparse.ArgumentParser(description='Visual Search')
parser.add_argument('-d','--data', metavar = 'DIR',help = 'path to dataset')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('-m','--model', nargs='+',default = [],
                    help='model file name, architecture, and path to latest checkpoint (ex: -m LandmarkNet_Land.py drn_38_d save/landmark/checkpoint.pth.tar)')
parser.add_argument('-b','--batch-size', type=int, default=64, metavar='N',
					help='input batch size for training (default: 64)')
parser.add_argument('-w','--workers', type = int, default = 16, metavar = 'N',
					help='number of works for data londing')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
					help='number of epochs to train (default: 150)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
					help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
					help='learning rate (default: 0.01)')
parser.add_argument('-p','--print-freq', default=100, type=int, metavar='N', 
					help='print frequency (default: 100)')

def main():
	#default paramters load
	global args, best_r
	args = parser.parse_args()

	if args.cuda:
		cudnn.benchmark = True
		print("GPU Mode")
	else:
		print("CPU Mode")

	#MODEL LOADS
	models = MM.Model_Manage()
	
	#Landmark Models Load
	if args.model:
		try:
			model = models.add(args.model[0:2])
			model, model_params = models.chkpoint_load(model, ['epoch'], args.model[-1])
			if model_params is not None:
				args.start_epoch = model_params['epoch']

		except:
			print("Model Loading Error")
			return

	else:
		print("Can't load model")
		return

	if args.model[0] == 'LandmarkNet_Land.py':
		best_r = 1
		train = landmark.train
		validate = landmark.validate
		#Data loads
		datas = DM.Data_Manage(args.batch_size, IMG_SIZE = IMG_SIZE, workers = args.workers, CUDA = args.cuda)
		TD = datas.load('landmark_loader.py',True, root = args.data, is_train = True, clothes = 0)
		VD = datas.load('landmark_loader.py',False, root = args.data, is_train = False, clothes = 0)

		criterion = nn.SmoothL1Loss().cuda()
		optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

	elif args.model[0] == 'MultiLabelNet.py':
		best_r = 0
		train = attribute.train
		validate = attribute.validate
		#Data loads
		datas = DM.Data_Manage(args.batch_size, IMG_SIZE = IMG_SIZE, workers = args.workers, CUDA = args.cuda)
		TD = datas.load('multilabel_loader.py',True, root = args.data, is_train = True)
		VD = datas.load('multilabel_loader.py',False, root = args.data, is_train = False)
		
		criterion = [nn.MultiLabelSoftMarginLoss().cuda(), nn.CrossEntropyLoss().cuda()]
		optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

	else:
		print("Should input correct model file name")
		return


	for epoch in range(args.start_epoch, args.epochs):
		optimizer = utility.adjust_learning_rate(optimizer, args.lr, epoch)
		train(TD, model, criterion, optimizer, epoch, args.print_freq)
		r = validate(VD, model, args.print_freq)
		is_best = r > best_r
		best_r = max(r, best_r)
		utility.save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.model[1],
			'state_dict': model.state_dict(),
			'best': best_r,
		}, is_best,
		filename = str(args.model[0])+"_"+str(args.model[1])+"_"+str(IMG_SIZE)+".pth.tar")

	return

if __name__ == '__main__':
	main()    
