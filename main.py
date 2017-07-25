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
from image_loader import ImageFolder
from VisualNet import vsnet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Visual Shopping')
parser.add_argument('data', metavar = 'DIR',help = 'path to dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
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
parser.add_argument('--anchor', default='', type=str,
					help='path to anchor image folder')
parser.add_argument('--feature-size', default=1000, type=int,
					help='fully connected layer size')
parser.add_argument('--save-db', action='store_true', default=False, 
					help='save inferencing result to mongodb')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')

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
    
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	#Annotation 로딩
	#Landmark
	landmark_dataframe = pd.read_table(data_path+"/Anno/list_landmarks_inshop.txt",sep="\s+")
	
	#이미지 로딩
	image_data = torch.utils.data.DataLoader(
		ImageFolder(data_path,landmark_dataframe,transforms.Compose([
			transforms.Scale(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size = args.batch_size,
		shuffle = False,
		num_workers = args.workers, 
		pin_memory = pin,
	)
	print("Complete Data loading(%s)" % len(image_data))
	model = vsnet()
	if args.cuda:
		model = torch.nn.DataParallel(model).cuda()

# = nn.L1Loss().cuda()
	criterion = nn.MSELoss().cuda()
	optimizer = torch.optim.Adagrad(model.parameters(), args.lr,weight_decay=args.weight_decay)

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)
		# train for one epoch
		train(image_data, model, criterion, optimizer, epoch)
		# evaluate on validation set
		#prec1 = validate(val_loader, model, criterion)

	    # remember best prec@1 and save checkpoint
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch, 
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
		}, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	# switch to train mode
	model.train()
	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target).float().cuda(async=True)
		# compute output
		output_1,output_2,output_3,output_4 = model(input_var)
#vis_output = output.narrow(1,0,2)
#		loc_output = output.narrow(1,8,24)

#		vis_target = target_var.narrow(1,0,2)
		target_1 = target_var.narrow(1,0,6)
		target_2 = target_var.narrow(1,6,12)
		target_3 = target_var.narrow(1,12,18)
		target_4 = target_var.narrow(1,18,24)

		loss = criterion(output_1, target_1)
		#lloss = loc_loss(loc_output, loc_target)
		
		# measure accuracy and record loss
		#prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
		losses.update(loss.data[0], input.size(0))
		#top1.update(prec1[0], input.size(0))
		#top5.update(prec5[0], input.size(0))

	    # compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		#lloss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1))


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
	lr = args.lr * (0.1 ** (epoch // 15))
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
