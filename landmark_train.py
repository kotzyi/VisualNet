'''To-Do
0. Visual Loss와 Location Loss의 통합 필요. 현재는 노멀라이즈 되어있지 않기 때문에, Visual Loss가 손해보는 구조임.
1. Triplet Loss 구성하기위한 새로운 네트워크 개발 필요 
2. 실제 triplet을 구하기 위한 새로우 main 개발 필요
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
import numpy as np
import readline
from glob import glob
import pymongo
from pymongo import MongoClient
import pandas as pd
from pandas import Series, DataFrame
from landmark_loader import ImageFolder
from LandmarkNet import landmarknet
import math

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Visual Shopping')
parser.add_argument('data', metavar = 'DIR',help = 'path to dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                    help='learning rate (default: 0.002)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Visual_Search', type=str,
                    help='name of experiment')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--workers', type = int, default = 8, metavar = 'N',
					help='number of works for data londing')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,	metavar='W', 
					help='weight decay (default: 1e-4)')
parser.add_argument('--save-db', action='store_true', default=False, 
					help='save inferencing result to mongodb')
parser.add_argument('--print-freq', '-p', default=100, type=int,
					metavar='N', help='print frequency (default: 100)')
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
	
	model = landmarknet()
	if args.cuda:
		model = torch.nn.DataParallel(model).cuda()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			#best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	#이미지 로딩
	image_data = torch.utils.data.DataLoader(
		ImageFolder(data_path,transforms.Compose([
			transforms.Scale(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size = args.batch_size,
		shuffle = True,
		num_workers = args.workers, 
		pin_memory = pin,
	)
	val_data = torch.utils.data.DataLoader(
		ImageFolder("/data/deep-fashion/in-shop/",transforms.Compose([
			transforms.Scale(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size = args.batch_size,
		shuffle = True,
		num_workers = args.workers,
		pin_memory = pin,
	)

	print("Complete Data loading(%s)" % len(image_data))

	softmax = nn.CrossEntropyLoss().cuda()
	l1loss = nn.SmoothL1Loss().cuda()
	#Convolution값 고정
	params = filter(lambda p: p.requires_grad, model.parameters())
	#params = model.parameters()
	optimizer = torch.optim.Adagrad(params, args.lr,weight_decay=args.weight_decay)

	if args.evaluate:
		validate(image_data, model, softmax,l1loss)
		return


	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)
		# train for one epoch
		train(image_data, model, softmax, l1loss, optimizer, epoch)
		# evaluate on validation set
		validate(val_data, model, softmax,l1loss)

	    # remember best prec@1 and save checkpoint
		#		is_best = prec1 > best_prec1
		#		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': 'vgg16', 
			'state_dict': model.state_dict(),
		#'best_prec1': best_prec1,
		#}, is_best)
		},True)

def validate(val_loader, model, softmax, l1loss):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses_v = AverageMeter()
	losses_l = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	# switch to train mode
    
	model.eval()
	end = time.time()
	for i, (input, clothes_type, collar, sleeve, waistline, hem, path) in enumerate(val_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		input_var = Variable(input)
		# compute output
		vis_collar, vis_sleeve, vis_waistline, vis_hem, collar_out, sleeve_out, waistline_out, hem_out = model(input_var)

	    # Answers
		collar = Variable(collar).float().cuda(async=True)
		sleeve = Variable(sleeve).float().cuda(async=True)
		waistline = Variable(waistline).float().cuda(async=True)
		hem = Variable(hem).float().cuda(async=True)

		hem_vis_loss = softmax(vis_hem[:,0:3], hem[:,0].long()) + softmax(vis_hem[:,3:6], hem[:,1].long())
		hem_land_loss = l1loss(hem_out[:,0:2], hem[:,2:4]) + l1loss(hem_out[:,2:4], hem[:,4:6])

		collar_vis_loss = softmax(vis_collar[:,0:3], collar[:,0].long()) + softmax(vis_collar[:,3:6], collar[:,1].long())
		collar_land_loss =l1loss(collar_out[:,0:2], collar[:,2:4]) + l1loss(collar_out[:,2:4], collar[:,4:6])

		sleeve_vis_loss = softmax(vis_sleeve[:,0:3], sleeve[:,0].long()) + softmax(vis_sleeve[:,3:6], sleeve[:,1].long())
		sleeve_land_loss = l1loss(sleeve_out[:,0:2], sleeve[:,2:4]) + l1loss(sleeve_out[:,2:4], sleeve[:,4:6])

		waistline_vis_loss = softmax(vis_waistline[:,0:3], waistline[:,0].long()) + softmax(vis_waistline[:,3:6], waistline[:,1].long())
		waistline_land_loss = l1loss(waistline_out[:,0:2], waistline[:,2:4]) + l1loss(waistline_out[:,2:4], waistline[:,4:6])

		vis_loss = hem_vis_loss + collar_vis_loss + sleeve_vis_loss + waistline_vis_loss
		land_loss = (hem_land_loss + collar_land_loss + sleeve_land_loss + waistline_land_loss)
		loss = land_loss + vis_loss

		#print("IN: ",sleeve[0])
		#print("OUT: ",sleeve_out[0])

		# measure accuracy and record loss
		prec1, _ = accuracy(vis_hem.data[:,0:3], hem.long().data[:,0].contiguous(), topk=(1, 1))
		losses_v.update(vis_loss.data[0], input.size(0))
		losses_l.update(land_loss.data[0], input.size(0))
		top1.update(prec1[0], input.size(0))
		#top5.update(prec5[0], input.size(0))

        # measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		if i % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
				'Visibility Loss {loss_v.val:.4f} ({loss_v.avg:.4f})\t'
				'Location Loss {loss_l.val:.4f} ({loss_l.avg:.4f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i, len(val_loader), loss_v=losses_v, loss_l = losses_l, top1=top1))

def train(train_loader, model, softmax, l1loss, optimizer, epoch):
	'''
	In clothes type, 
	"1" represents upper-body clothes, 
	"2" represents lower-body clothes, 
	"3" represents full-body clothes. 
	Upper-body clothes possess six fahsion landmarks, 
	lower-body clothes possess four fashion landmarks, 
	full-body clothes possess eight fashion landmarks.
	For upper-body clothes, landmark annotations are listed in the order of ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]
	For lower-body clothes, landmark annotations are listed in the order of ["left waistline", "right waistline", "left hem", "right hem"]
	For upper-body clothes, landmark annotations are listed in the order of ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"].
	'''
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses_v = AverageMeter()
	losses_l = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	# switch to train mode
	model.train()
	end = time.time()
	for i, (input, clothes_type, collar, sleeve, waistline, hem, path) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		input_var = Variable(input)
	
		#for name,m in model.named_modules():
		#	if name in ['module.features.28']:
		#		print(m.weight)
		# compute output
		vis_collar, vis_sleeve, vis_waistline, vis_hem, collar_out, sleeve_out, waistline_out, hem_out = model(input_var)
	
		# Answers
		collar = Variable(collar).float().cuda(async=True)
		sleeve = Variable(sleeve).float().cuda(async=True)
		waistline = Variable(waistline).float().cuda(async=True)
		hem = Variable(hem).float().cuda(async=True)

		hem_vis_loss = softmax(vis_hem[:,0:3], hem[:,0].long()) + softmax(vis_hem[:,3:6], hem[:,1].long())
		hem_land_loss = l1loss(hem_out[:,0:2], hem[:,2:4]) + l1loss(hem_out[:,2:4], hem[:,4:6])

		collar_vis_loss = softmax(vis_collar[:,0:3], collar[:,0].long()) + softmax(vis_collar[:,3:6], collar[:,1].long())
		collar_land_loss =l1loss(collar_out[:,0:2], collar[:,2:4]) + l1loss(collar_out[:,2:4], collar[:,4:6])

		sleeve_vis_loss = softmax(vis_sleeve[:,0:3], sleeve[:,0].long()) + softmax(vis_sleeve[:,3:6], sleeve[:,1].long())
		sleeve_land_loss = l1loss(sleeve_out[:,0:2], sleeve[:,2:4]) + l1loss(sleeve_out[:,2:4], sleeve[:,4:6])

		waistline_vis_loss = softmax(vis_waistline[:,0:3], waistline[:,0].long()) + softmax(vis_waistline[:,3:6], waistline[:,1].long())
		waistline_land_loss = l1loss(waistline_out[:,0:2], waistline[:,2:4]) + l1loss(waistline_out[:,2:4], waistline[:,4:6])
		
		vis_loss = hem_vis_loss + collar_vis_loss + sleeve_vis_loss + waistline_vis_loss
		land_loss = hem_land_loss + collar_land_loss + sleeve_land_loss + waistline_land_loss
		loss = land_loss + vis_loss
		# measure accuracy and record loss
		prec1, _ = accuracy(vis_hem.data[:,0:3], hem.long().data[:,0].contiguous(), topk=(1, 1))
		losses_v.update(vis_loss.data[0], input.size(0))
		losses_l.update(land_loss.data[0], input.size(0))
		top1.update(prec1[0], input.size(0))
		#top5.update(prec5[0], input.size(0))
		#print("IN:",sleeve[0])
		#print("OUT:",sleeve_out[0])
	    # compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Visibility Loss {loss_v.val:.4f} ({loss_v.avg:.4f})\t'
				'Location Loss {loss_l.val:.4f} ({loss_l.avg:.4f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
				data_time=data_time, loss_v=losses_v, loss_l = losses_l, top1=top1))


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total:
		print()

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
	w12 = torch.sum(x1 * x2, dim)
	w1 = torch.norm(x1, 2, dim)
	w2 = torch.norm(x2, 2, dim)
	return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')

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
	lr = args.lr * (0.9 ** (epoch // 20))
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

if __name__ == '__main__':
	main()    
