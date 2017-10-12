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
from data.landmark_loader import ImageFolder
import random
from utils.drawing import Drawing
from PIL import Image
import socket
import struct
import cv2
from models.LandmarkNet_Deep import Res_Deeplab
from utils.loss import CrossEntropy2d

TCP_IP = '10.214.35.36'
TCP_PORT = 5005
BUFFER_SIZE = 1024
RECV_PATH = '/home/jlee/VisualNet/received_img.jpg'
IMG_SIZE = 224
IGNORE_LABEL = -1

# Training settings
parser = argparse.ArgumentParser(description='Visual Search')
parser.add_argument('data', metavar = 'DIR',help = 'path to dataset')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=2.5e-4, metavar='LR',
                    help='learning rate (default: 2.5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--workers', type = int, default = 8, metavar = 'N',
					help='number of works for data londing')
parser.add_argument('--wd', default=0.0005, type=float, metavar='WD',
					help='weight decay (default:0.0005)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('-a', '--arch', default='resnet50', type=str,
					help='Architecture of network for training in resnet18,34,50,101,152')
parser.add_argument('-c', '--clothes', default=0, type=int,
					help='Clothes type:0 - upper body, 1 - lower body, 2 - full body')
parser.add_argument('-d', '--draw', action='store_true', default = False,
					help='drawing landmark points on images')
parser.add_argument('-t', '--test', action='store_true', default = False,
					help='Connecting client for image receving')

def main():
	#기본 설정 부분
	global args, best_acc
	args = parser.parse_args()
	data_path = args.data
	args.cuda = not args.no_cuda and torch.cuda.is_available()
#torch.manual_seed(random.randint(1, 10000))
	pin = True
	best_dist = 100.0

	if args.cuda:
		print("GPU Mode")
		#torch.cuda.manual_seed(random.randint(1, 10000))
	else:
		pin = False
		print("CPU Mode")
	
	model = Res_Deeplab(num_classes=7)

	if args.cuda:
		model = torch.nn.DataParallel(model).cuda()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			args.clothes = checkpoint['clothes_type']
			best_dist = checkpoint['best_distance']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	trans = transforms.Compose([
					transforms.Scale(IMG_SIZE),
					transforms.CenterCrop(IMG_SIZE),
					transforms.ToTensor(),
					normalize,
	])
	val_data = torch.utils.data.DataLoader(
		ImageFolder(data_path,False,args.clothes,trans),
		batch_size = args.batch_size,
		shuffle = True,
		num_workers = args.workers,
		pin_memory = pin,
	)
	print("Complete Validation Data loading(%s)" % len(val_data))
	
	#TEST

	if args.evaluate:
		if args.test:
			model.eval()
			d = Drawing()
#while True:
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.bind((TCP_IP, TCP_PORT))
			sock.listen(1000)
			print("Listening...")

			while True:
				client, addr = sock.accept()
				print("Got connection from",addr)
				print("Receiving...")

				recv_file(client)

				input = Image.open(RECV_PATH).convert('RGB')
				input_tensor = torch.unsqueeze(trans(input),0)

				input_var = Variable(input_tensor).cuda()

				collar_out, sleeve_out, hem_out, _ = model(input_var)
				c = collar_out.data.cpu().tolist()[0]
				s = sleeve_out.data.cpu().tolist()[0]
				h = hem_out.data.cpu().tolist()[0]

				width,height = input.size
				points = [c[0]*width,c[1]*height,c[2]*width,c[3]*height,
						s[0]*width,s[1]*height,s[2]*width,s[3]*height,
						h[0]*width,h[1]*height,h[2]*width,h[3]*height]

				img = cv2.imread(RECV_PATH)

				for i, landmark in enumerate(zip(points[::2], points[1::2])):
					if landmark[0] > 0 and landmark[1] > 0:
						img = cv2.circle(img, (int(landmark[0]),int(landmark[1])), 5, (0,0,255), -1)

				cv2.imwrite(RECV_PATH, img)

				print("Drawing Done")

				send_file(client)
				client.close()
				
		else:
			validate(val_data, model, args.clothes)

		return

	image_data = torch.utils.data.DataLoader(
		ImageFolder(data_path,True,args.clothes,trans),
		batch_size = args.batch_size,
		shuffle = True,
		num_workers = args.workers, 
		pin_memory = pin,
	)

	print("Complete Data loading(%s)" % len(image_data))

	#criterion = nn.L1Loss().cuda()
	params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = torch.optim.Adagrad(params, lr = args.lr, weight_decay = args.wd)
	for epoch in range(args.start_epoch, args.epochs):
		# train for one epoch
		train(image_data, model, optimizer, epoch, args.clothes)
		# evaluate on validation set
		validate(val_data, model, args.clothes)
	    # remember best prec@1 and save checkpoint
#is_best = dist < best_dist
#		best_dist = min(dist, best_dist)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'state_dict': model.state_dict(),
		}, True)

def validate(val_loader, model, clothes_type):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	distance = AverageMeter()
    
	model.eval()
	end = time.time()
	d = Drawing()
	interp = nn.Upsample(size=(IMG_SIZE,IMG_SIZE), mode='bilinear')

	if clothes_type == 0:
		for i, (input, collar, sleeve, hem, path) in enumerate(val_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			input_var = Variable(input)

			pred = interp(model(input_var))
			# drawing landmark point on images
			"""
			if args.draw:
				for img_path, c, s, h in zip(path,collar_list,sleeve_list,hem_list):
					img = Image.open(args.data + img_path)
					width,height = img.size
					d.draw_landmark_point(args.data + img_path, [c[0]*width,c[1]*height,c[2]*width,c[3]*height,
																s[0]*width,s[1]*height,s[2]*width,s[3]*height,
																h[0]*width,h[1]*height,h[2]*width,h[3]*height])
			"""
	        # Answers
			#dist = accuracy((collar[:,2:6],sleeve[:,2:6],hem[:,2:6]),(collar_out,sleeve_out,hem_out))
			#distance.update(dist.data[0], 1)
			#print(path[0],sleeve[0],sleeve_out[0])
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			if i % args.print_freq == 0:
				print('Epoch: [{0}/{1}]\t'
						'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						'Data {data_time.val:.3f} ({data_time.avg:.3f})'
						.format(i, len(val_loader), batch_time=batch_time,data_time=data_time))

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

def train(train_loader, model, optimizer, epoch, clothes_type):
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
	losses = AverageMeter()
	distance = AverageMeter()
	# switch to train mode
	model.train()

	adjust_learning_rate(optimizer, epoch)
	
	interp = nn.Upsample(size=(IMG_SIZE,IMG_SIZE), mode='bilinear')
	end = time.time()
	if clothes_type == 0:
		for i, (input, collar, sleeve, hem, path) in enumerate(train_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			input_var = Variable(input)

			pred = interp(model(input_var))
			optimizer.zero_grad()
			labels = make_labels([collar,sleeve,hem])

			loss = loss_calc(pred, labels)
			# Answers
			"""
			collar = Variable(collar).float().cuda(async=True)
			sleeve = Variable(sleeve).float().cuda(async=True)
			hem = Variable(hem).float().cuda(async=True)

			hem_loss = criterion(hem_out, hem[:,2:6])
			collar_loss =criterion(collar_out, collar[:,2:6])
			sleeve_loss = criterion(sleeve_out, sleeve[:,2:6])

			loss = hem_loss + collar_loss + sleeve_loss

			dist = accuracy((collar[:,2:6],sleeve[:,2:6],hem[:,2:6]),(collar_out,sleeve_out,hem_out))
			losses.update(loss.data[0], input.size(0))
			distance.update(dist.data[0], 1)
			"""

			losses.update(loss.data[0], input.size(0))
			# compute gradient and do SGD step
			loss.backward()
			optimizer.step()
			

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			if i % args.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
						'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
						'Location Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time, loss = losses))

	elif clothes_type == 1:
		for i, (input, waistline, hem, path) in enumerate(train_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			input_var = Variable(input)
			optimizer.zero_grad()
			waistline_out, hem_out, _ = model(input_var)

			# Answers
			waistline = Variable(waistline).float().cuda(async=True)
			hem = Variable(hem).float().cuda(async=True)

			hem_loss = criterion(hem_out, hem[:,2:6])
			waistline_loss = criterion(waistline_out, waistline[:,2:6])

			loss = hem_loss + waistline_loss

			dist = accuracy((waistline[:,2:6],hem[:,2:6]),(waistline_out,hem_out))
			losses.update(loss.data[0], input.size(0))
			distance.update(dist.data[0], 1)

	        # compute gradient and do SGD step
			loss.backward()
			optimizer.step()

	        # measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
						'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
						'Location Loss {loss.val:.4f} ({loss.avg:.4f})\t'
						'Distances {dist.val:.4f} ({dist.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss = losses, dist = distance))

	else:
		for i, (input, collar, sleeve, waistline, hem, path) in enumerate(train_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			input_var = Variable(input)
			optimizer.zero_grad()
			#for name,m in model.named_modules():
			#	if name in ['module.features.28']:
			#		print(m.weight)
			# compute output
			collar_out, sleeve_out, waistline_out, hem_out, _ = model(input_var)
	
			# Answers
			collar = Variable(collar).float().cuda(async=True)
			sleeve = Variable(sleeve).float().cuda(async=True)
			waistline = Variable(waistline).float().cuda(async=True)
			hem = Variable(hem).float().cuda(async=True)

			hem_loss = criterion(hem_out, hem[:,2:6])
			collar_loss =criterion(collar_out, collar[:,2:6])
			sleeve_loss = criterion(sleeve_out, sleeve[:,2:6])
			waistline_loss = criterion(waistline_out, waistline[:,2:6])
		
			loss = hem_loss + collar_loss + sleeve_loss + waistline_loss

			dist = accuracy((collar[:,2:6],sleeve[:,2:6],waistline[:,2:6],hem[:,2:6]),(collar_out,sleeve_out,waistline_out,hem_out))
			losses.update(loss.data[0], input.size(0))
			distance.update(dist.data[0], 1)
	    
			# compute gradient and do SGD step
			loss.backward()
			optimizer.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					'Location Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Distances {dist.val:.4f} ({dist.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss = losses, dist = distance))

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
	lr = args.lr * (0.5 ** (epoch // 10))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

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

def recv_file(client):
	with open(RECV_PATH,'wb') as f:
		l = client.recv(BUFFER_SIZE)
		
		while (l):
			f.write(l)
			l=client.recv(BUFFER_SIZE)
	
	print("Done Receiving")
#client.close()

	return f
		
def send_file(client):
	with open(RECV_PATH, 'rb') as f:
		l = f.read(BUFFER_SIZE)

		while (l):
			client.send(l)
			l = f.read(BUFFER_SIZE)

	client.shutdown(socket.SHUT_WR)
	print("Done Sending")

def make_labels(raw):
	target = torch.zeros(len(raw[0]),IMG_SIZE,IMG_SIZE)
	target.fill_(6)

	for coords in raw:
		i = 0
		for j,coord in enumerate(coords):
			if coord[0] == 0: # 주의! 찾을 수 없음이 255가 아닌지 살펴 볼 것!
				x = int(coord[2]*IMG_SIZE)
				y = int(coord[3]*IMG_SIZE)
				if IMG_SIZE == x:
					x = IMG_SIZE - 1
				if IMG_SIZE == y:
					y = IMG_SIZE - 1

				target[j][x][y] = i
			if coord[1] == 0:
				x = int(coord[4]*IMG_SIZE)
				y = int(coord[5]*IMG_SIZE)
				if IMG_SIZE == x:
					x = IMG_SIZE - 1
				if IMG_SIZE == y:
					y = IMG_SIZE - 1

				target[j][x][y] = i + 1

		i += 2

	return target

def loss_calc(pred, label):
	# out shape batch_size x channels x h x w -> batch_size x channels x h x w
	# label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
	label = Variable(label.long()).cuda()
	criterion = CrossEntropy2d().cuda()
	
	return criterion(pred, label)


if __name__ == '__main__':
	main()    