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
from torch.autograd import Variable

import utils.model_manage as MM
import utils.communicate as comm
import utils.data_manage as DM
import utils.utility as utility
IMG_SIZE = 320
WINDOW_SIZE = 3

# Training settings
parser = argparse.ArgumentParser(description='Visual Search')
parser.add_argument('-d','--data', metavar = 'DIR',help = 'path to dataset')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('-m','--model', nargs='+',default = [],
                    help='model file name, architecture, and path to latest checkpoint (ex: -l LandmarkNet_Land.py drn_38_d save/landmark/checkpoint.pth.tar)')
parser.add_argument('-b','--batch-size', type=int, default=64, metavar='N',
					help='input batch size for training (default: 64)')
parser.add_argument('-w','--workers', type = int, default = 16, metavar = 'N',
					help='number of works for data londing')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
					help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
					help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
					help='learning rate (default: 0.002)')

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

	if args.cuda:
		print("GPU Mode")
	else:
		print("CPU Mode")

	#MODEL LOADS
	models = MM.Model_Manage()
	
	#Landmark Models Load
	if args.model:
		model = models.add(args.model[0:2])
		model, _ = models.chkpoint_load(model, ['epoch'], args.model[-1])
		if model is not None:
			model.train()

	else:
		print("Can't load model")
		return

	#DATA LOADS
	datas = DM.Data_Manage(args.batch_size, IMG_SIZE = IMG_SIZE, workers = args.workers, CUDA = args.cuda)
	TD = datas.load('landmark_loader.py',True, root = args.data, isTrain = True, clothes = 0)

	criterion = nn.L1Loss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)


	for epoch in range(args.start_epoch, args.epochs):
		optimizer = utility.adjust_learning_rate(optimizer, args.lr, epoch)
		train(TD, model, criterion, optimizer, epoch)
		dist = validate(TD, model)
		is_best = dist < best_dist
		best_dist = min(dist, best_dist)
		utility.save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.model[1],
			'state_dict': model.state_dict(),
			'best_distance': best_dist,
		}, is_best,
		filename = "_".join(str(args.model))+"_"+str(args.data))


if __name__ == '__main__':
	main()    
