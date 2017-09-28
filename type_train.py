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
import torchvision.models as models
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from data.type_loader import ImageFolder
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
parser.add_argument('-n', '--num_classes', default=3, type=int,
					help='The number of cloth types')

def main():
	#기본 설정 부분
	global args, best_acc
	args = parser.parse_args()
	data_path = args.data
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	torch.manual_seed(random.randint(1, 10000))
	pin = True
	best_acc = 0.0

	if args.cuda:
		print("GPU Mode")
		torch.cuda.manual_seed(random.randint(1, 10000))
	else:
		pin = False
		print("CPU Mode")
	
	model = models.__dict__[args.arch](pretrained=True)
	model.fc = nn.Linear(512, args.num_classes)

	if args.cuda:
		model = torch.nn.DataParallel(model).cuda()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_acc = checkpoint['best_acc']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	
	val_data = torch.utils.data.DataLoader(
		ImageFolder(data_path,False,transforms.Compose([
			transforms.Scale(400),
			transforms.CenterCrop(400),
			transforms.RandomHorizontalFlip(),
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
		validate(val_data, model)
		return

	image_data = torch.utils.data.DataLoader(
		ImageFolder(data_path,True,transforms.Compose([
			transforms.Scale(400),
			transforms.CenterCrop(400),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size = args.batch_size,
		shuffle = True,
		num_workers = args.workers, 
		pin_memory = pin,
	)

	print("Complete Data loading(%s)" % len(image_data))

	criterion = nn.CrossEntropyLoss().cuda()
	params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = torch.optim.Adagrad(params, lr = args.lr, weight_decay = args.weight_decay)

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)
		# train for one epoch
		train(image_data, model, criterion, optimizer, epoch)
		# evaluate on validation set
		acc = validate(val_data, model)

	    # remember best prec@1 and save checkpoint
		is_best = acc > best_acc
		best_acc = max(acc, best_acc)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch, 
			'state_dict': model.state_dict(),
			'best_acc': best_acc,
		}, is_best,args.arch+'_'+'type.pth.tar')

def validate(val_loader, model):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
    
	model.eval()
	end = time.time()
	for i, (input, target) in enumerate(val_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target)

		# compute output
		output = model(input_var)
		
		prec1, _ = accuracy(output.data, target, topk=(1, 1))
		top1.update(prec1[0], input.size(0))
        # measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
		    print('Epoch: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					'Accuracy {acc.val:.4f} ({acc.avg:.4f})'.format(i, len(val_loader), batch_time=batch_time,data_time=data_time, acc = top1))
	print('Test Result: Distances ({acc.avg:.4f})'.format(acc = top1))

	return top1.avg

def train(train_loader, model, criterion, optimizer, epoch):
	'''
	In clothes type, 
	"1" represents upper-body clothes, 
	"2" represents lower-body clothes, 
	"3" represents full-body clothes. 
	'''
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	# switch to train mode
	model.train()
	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target)
	
		output = model(input_var)
		loss = criterion(output, target_var)

		prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
	
		top1.update(prec1[0], input.size(0))
		losses.update(loss.data[0], input.size(0))
	    
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
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
						epoch, i, len(train_loader), batch_time=batch_time,
						data_time=data_time, loss=losses, top1=top1))

def save_checkpoint(state, is_best, filename='type.pth.tar'):
	args = parser.parse_args()
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'typenet_best.pth.tar')

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
	lr = args.lr * (0.5 ** (epoch // 10))
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
