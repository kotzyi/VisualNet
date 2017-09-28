'''To-Do
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
from data.image_loader import ImageFolder
from models.VisualNet import visualnet
import random
# Training settings
parser = argparse.ArgumentParser(description='Visual Search')
parser.add_argument('data', metavar = 'DIR',help = 'path to dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                    help='learning rate (default: 0.002)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--workers', type = int, default = 8, metavar = 'N',
					help='number of works for data londing')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,	metavar='W', 
					help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
					metavar='N', help='print frequency (default: 100)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('-a', '--arch', default='resnet50', type=str,
					help='Architecture of network for training in resnet18,34,50,101,152')
parser.add_argument('-c', '--clothes', default=0, type=int,
					help='Clothes type:0 - upper body, 1 - lower body, 2 - full body')

def main():
	#기본 설정 부분
	global args, best_acc
	args = parser.parse_args()
	data_path = args.data
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	torch.manual_seed(random.randint(1, 10000))
	pin = True
	best_dist = 100.0

	if args.cuda:
		print("GPU Mode")
		torch.cuda.manual_seed(random.randint(1, 10000))
	else:
		pin = False
		print("CPU Mode")
	
	model = visualnet(args.arch, args.clothes)
	if args.cuda:
		model = torch.nn.DataParallel(model).cuda()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			args.clothes = checkpoint['clothes_type']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	
	val_data = torch.utils.data.DataLoader(
		ImageFolder(data_path,transforms.Compose([
			transforms.Scale(256),
			transforms.CenterCrop(256),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size = args.batch_size,
		shuffle = True,
		num_workers = args.workers,
		pin_memory = pin,
	)

	print("Complete Validation Data loading(%s)" % len(val_data))

	if args.evaluate:
		validate(val_data, model, args.clothes)
		return

def validate(val_loader, model, clothes_type):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	distance = AverageMeter()
    
	model.eval()
	end = time.time()

	if clothes_type == 0:
		for i, (input, path) in enumerate(val_loader):
			data_time.update(time.time() - end)
			input_var = Variable(input)

			# compute output
			collar_out, sleeve_out, hem_out, _ = model(input_var)

	        # Answers
			collar = Variable(collar).float().cuda(async=True)
			sleeve = Variable(sleeve).float().cuda(async=True)
			hem = Variable(hem).float().cuda(async=True)
			dist = accuracy((collar[:,2:6],sleeve[:,2:6],hem[:,2:6]),(collar_out,sleeve_out,hem_out))
			distance.update(dist.data[0], 1)
			print(path[0],sleeve[0],sleeve_out[0])
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Epoch: [{0}/{1}]\t'
						'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
						'Distances {dist.val:.4f} ({dist.avg:.4f})'.format(i, len(val_loader), batch_time=batch_time,data_time=data_time, dist = distance))

	elif clothes_type == 1:
		for i, (input, waistline, hem, path) in enumerate(val_loader):
			data_time.update(time.time() - end)
			input_var = Variable(input)
			
			waistline_out, hem_out, _ = model(input_var)
			waistline = Variable(waistline).float().cuda(async=True)
			hem = Variable(hem).float().cuda(async=True)
			print(path[0],waistline[0],waistline_out[0])
			dist = accuracy((waistline[:,2:6],hem[:,2:6]),(waistline_out,hem_out))
			distance.update(dist.data[0], 1)

			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Epoch: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					'Distances {dist.val:.4f} ({dist.avg:.4f})'.format(i, len(val_loader), batch_time=batch_time,data_time=data_time, dist = distance))

	else:
		for i, (input, collar, sleeve, waistline, hem, path) in enumerate(val_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			input_var = Variable(input)
			# compute output
			collar_out, sleeve_out, waistline_out, hem_out, _ = model(input_var)

			# Answers
			collar = Variable(collar).float().cuda(async=True)
			sleeve = Variable(sleeve).float().cuda(async=True)
			waistline = Variable(waistline).float().cuda(async=True)
			hem = Variable(hem).float().cuda(async=True)

			dist = accuracy((collar[:,2:6],sleeve[:,2:6],waistline[:,2:6],hem[:,2:6]),(collar_out,sleeve_out,waistline_out,hem_out))
			distance.update(dist.data[0], 1)
		

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Epoch: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					'Distances {dist.val:.4f} ({dist.avg:.4f})'.format(i, len(val_loader), batch_time=batch_time,data_time=data_time, dist = distance))
	
	print('Test Result: Distances ({dist.avg:.4f})'.format(dist = distance))
	return distance.avg


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


def accuracy(target,output):
	"""Computes the precision@k for distances of the landmarks"""
	dist_function = nn.PairwiseDistance(p=1)
	
	distances = dist_function(torch.cat(target,1),torch.cat(output,1))
	
	if args.clothes == 0:
		return distances.mean() /12
	elif args.clothes == 1:
		return distances.mean() / 8
	else:
		return distances.mean() / 16

if __name__ == '__main__':
	main()    
