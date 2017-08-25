'''To-Do
clear 64x8x512x4x4 의 스택으로 이루어진 데이터 셋 만들 것
0. Triplet loader 작성할 것.,!
'''

from __future__ import print_function
import argparse
import os
import sys
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from data.triplet_loader import TripletFolder
from models.LandmarkNet import landmarknet
from models.VisualNet import visualnet
import math

image_size = 256
conv_size = 28
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Trainer for Triplet Loss of Rich Annotations')
parser.add_argument('data', metavar = 'DIR',help = 'path to dataset')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--land-load', default='', type=str,
                    help='path to latest checkpoint of posenet(default: none)')
parser.add_argument('--name', default='Visual_Search', type=str,
                    help='name of experiment')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--workers', type = int, default = 8, metavar = 'N',
					help='number of works for data londing')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,	metavar='W', 
					help='weight decay (default: 1e-4)')
parser.add_argument('--anchor', default='', type=str,
					help='path to anchor image folder')
parser.add_argument('--feature-size', default=1000, type=int,
					help='fully connected layer size')
parser.add_argument('--save-db', action='store_true', default=False, 
					help='save inferencing result to mongodb')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')

best_acc = 0

def main():
	#기본 설정 부분
	global args, best_acc
	args = parser.parse_args()
	data_path = args.data
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	torch.manual_seed(args.seed)
	pin = True
	if args.cuda:
		print("GPU Mode")
		torch.cuda.manual_seed(args.seed)
	else:
		pin = False
		print("CPU Mode")
	
	land_model = landmarknet()
	visual_model = visualnet()

	if args.cuda:
		land_model = torch.nn.DataParallel(land_model).cuda()
		visual_model = torch.nn.DataParallel(visual_model).cuda()
		triplet_loss = nn.TripletMarginLoss(margin=2.0, p=2).cuda()

	print("Model parameters loaded")
	
	# optionally resume from a checkpoint
	if args.land_load:
		if os.path.isfile(args.land_load):
			print("=> loading checkpoint '{}'".format(args.land_load))
			checkpoint = torch.load(args.land_load)
			#args.start_epoch = checkpoint['epoch']
			#best_prec1 = checkpoint['best_prec1']
			land_model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})".format(args.land_load, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.land_load))

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	#triplet data loading	
	triplet_train_data = torch.utils.data.DataLoader(
		TripletFolder(data_path,transforms.Compose([
			transforms.Scale(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=args.batch_size, 
		shuffle=True,
		num_workers = args.workers,
		pin_memory = pin,
	)

	print("Complete Data loading(%s)" % len(triplet_train_data))

	params = filter(lambda p: p.requires_grad, visual_model.parameters())
	optimizer = torch.optim.Adagrad(params, args.lr,weight_decay=args.weight_decay)

	if args.evaluate:
		validate(triplet_data, land_model, visual_model, triplet_loss, optimizer)
		return


	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)
		# train for one epoch
		train(epoch, triplet_train_data, land_model, visual_model, triplet_loss, optimizer)
		# evaluate on validation set
		#prec1 = validate(val_loader, model, criterion)

	    # remember best prec@1 and save checkpoint
		#		is_best = prec1 > best_prec1
		#		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': 'visualnet', 
			'state_dict': visual_model.state_dict(),
		#'best_prec1': best_prec1,
		#}, is_best)
		},True)

def train(epoch, triplet_data, land_model, visual_model, triplet_loss, optimizer):
	losses = AverageMeter()

	land_model.eval()
	visual_model.train()

	for i, (anchor, positive, negative) in enumerate(triplet_data):
		input_var = Variable(anchor)
		vis_collar, vis_sleeve, vis_waistline, vis_hem, collar_out, sleeve_out, waistline_out, hem_out, anchor_feature = land_model(input_var)
		anchor_pool5_layer = gen_pool5_layer((vis_collar[:,0:3], vis_collar[:,3:6],vis_sleeve[:,0:3], vis_sleeve[:,3:6],
				vis_waistline[:,0:3], vis_waistline[:,3:6],vis_hem[:,0:3], vis_hem[:,3:6]), 
				(collar_out[:,0:2], collar_out[:,2:4], sleeve_out[:,0:2], sleeve_out[:,2:4],
				 waistline_out[:,0:2], waistline_out[:,2:4], hem_out[:,0:2], hem_out[:,2:4]), anchor_feature)
		embedded_anchor = visual_model(anchor_feature, anchor_pool5_layer)

		input_var = Variable(positive)
		vis_collar, vis_sleeve, vis_waistline, vis_hem, collar_out, sleeve_out, waistline_out, hem_out, positive_feature = land_model(input_var)
		positive_pool5_layer = gen_pool5_layer((vis_collar[:,0:3], vis_collar[:,3:6],vis_sleeve[:,0:3], vis_sleeve[:,3:6],
				vis_waistline[:,0:3], vis_waistline[:,3:6],vis_hem[:,0:3], vis_hem[:,3:6]),
				(collar_out[:,0:2], collar_out[:,2:4], sleeve_out[:,0:2], sleeve_out[:,2:4],
				 waistline_out[:,0:2], waistline_out[:,2:4], hem_out[:,0:2], hem_out[:,2:4]), positive_feature)
		embedded_positive = visual_model(positive_feature, positive_pool5_layer)

		input_var = Variable(negative)
		vis_collar, vis_sleeve, vis_waistline, vis_hem, collar_out, sleeve_out, waistline_out, hem_out, negative_feature = land_model(input_var)
		negative_pool5_layer = gen_pool5_layer((vis_collar[:,0:3], vis_collar[:,3:6],vis_sleeve[:,0:3], vis_sleeve[:,3:6],
				vis_waistline[:,0:3], vis_waistline[:,3:6],vis_hem[:,0:3], vis_hem[:,3:6]),
				(collar_out[:,0:2], collar_out[:,2:4], sleeve_out[:,0:2], sleeve_out[:,2:4],
				 waistline_out[:,0:2], waistline_out[:,2:4], hem_out[:,0:2], hem_out[:,2:4]), negative_feature)
		embedded_negative = visual_model(negative_feature, negative_pool5_layer)	

		loss = triplet_loss(embedded_anchor, embedded_positive, embedded_negative)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
		losses.update(loss.data[0], anchor.size(0))

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'	
					'Location Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(triplet_data), loss=losses))

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total:
		print()

def save_checkpoint(state, is_best, filename='visual_checkpoint.pth.tar'):
	torch.save(state, filename)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def get_ROI(coord,feature_map):
	"""Get Region of Interest(Landmark) by using landmark coordinations"""
	pad = nn.ReflectionPad2d(2)
	
	f = pad(torch.unsqueeze(feature_map, 0))
	x = math.floor(coord[0])
	y = math.floor(coord[1])
	
	x_idx = Variable(torch.LongTensor([x+1,x+2,x+3,x+4]))
	y_idx = Variable(torch.LongTensor([y+1,y+2,y+3,y+4]))

	if args.cuda:
		x_idx = x_idx.cuda() # 4x4 patch cut
		y_idx = y_idx.cuda()

	f = torch.index_select(f, 3, x_idx)
	f = torch.index_select(f, 2, y_idx)
	
	return torch.squeeze(f)

def gen_pool5_layer(visualities, coords, feature):
	"""Generate pool5_layer map for training triplet network
	visualities, coords: 8 visual results that consist of args.batch_size x 3
	feature: args.batch_size x 512 x 28 x 28
	"""
	pool5_layer = 0 #Variable(torch.randn(8,512,4,4)).cuda()
	
	for i in range(len(feature)):
		pool5 = 0 #Variable(torch.randn(512,4,4)).cuda()
		
		for j,(vis, coord) in enumerate(zip(visualities, coords)):
			_, max_idx = torch.max(vis,1)
			
			if max_idx.data[0] == 0 and coord[i].data[0] > 0 and coord[i].data[1] > 0 and coord[i].data[0] < 256 and coord[i].data[1] < 256:
				f = get_ROI(coord[i].data / image_size * conv_size, feature[i])
			else:
				f = Variable(torch.zeros(512,4,4)).cuda()

			if j > 0:
				pool5 = torch.cat((pool5, f), 0)
			else:
				pool5 = f

		#Chunk and stack the pool5 > 8 x 512 x 4 x 4
		pool5 = torch.stack(torch.split(pool5,512,0),0)

		if i > 0:
			pool5_layer = torch.cat((pool5_layer, pool5), 0)
		else:
			pool5_layer = pool5

	pool5_layer = torch.stack(torch.split(pool5_layer,8,0),0)

	return pool5_layer

if __name__ == '__main__':
	main()    
