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
from models.LandmarkNet import landmarknet
import random
from utils.drawing import Drawing
from PIL import Image
import socket
import struct
import cv2
import numpy as np
import pickle

TCP_IP = '10.214.35.36'
TCP_PORT = 5005
BUFFER_SIZE = 4096
IMG_SIZE = 256
RECV_PATH = '/home/jlee/VisualNet/received_img.jpg'

# Training settings
parser = argparse.ArgumentParser(description='Visual Search')
parser.add_argument('data', metavar = 'DIR',help = 'path to dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.002)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
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
					help='Architecture of network for training in resnet18,34,50,101,152, drn 22d,38d,')
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
	else:
		pin = False
		print("CPU Mode")
	
	model = landmarknet(args.arch, args.clothes)
	if args.cuda:
		model = torch.nn.DataParallel(model).cuda()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
#args.start_epoch = checkpoint['epoch']
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
			sock.listen(10)
			print("Listening...")

			while True:
				conn, addr = sock.accept()
				print("Got connection from",addr)
				print("Receiving...")

				#input = recv_file(client)
				data = b""
				payload_size = struct.calcsize("I") 
				while True:
					while len(data) < payload_size:
						data += conn.recv(4096)
					packed_msg_size = data[:payload_size]
					data = data[payload_size:]
					msg_size = struct.unpack("I", packed_msg_size)[0]
					while len(data) < msg_size:
						data += conn.recv(4096)
					frame_data = data[:msg_size]
					data = data[msg_size:]
					frame = pickle.loads(frame_data)
					height, width = frame.shape[:2]

					input = Image.fromarray(frame)
					#input.save("img1.png","PNG")
					input_tensor = torch.unsqueeze(trans(input),0)
					
					input_var = Variable(input_tensor,volatile =True).cuda()
					collar_out, sleeve_out, hem_out, _ = model(input_var)
					
					c = collar_out.data.tolist()[0]
					s = sleeve_out.data.tolist()[0]
					h = hem_out.data.tolist()[0]

					coords=pickle.dumps(c+s+h)

					conn.send(coords)
#print(collar_out)
				"""
				while True:
					packet = conn.recv(4096)
					if not packet: 
						break
					data += packet
				"""
				frame = pickle.loads(frame_data)
				
				input = Image.fromarray(frame)
				#input = Image.open(frame).convert('RGB')
				#input = input.convert('RGB')
				input_tensor = torch.unsqueeze(trans(input),0)

				input_var = Variable(input_tensor,volatile =True).cuda()

				#TEST
				#print(interp(model_test(input_var)))

				collar_out, sleeve_out, hem_out, _ = model(input_var)
				c = collar_out.data.tolist()[0]
				s = sleeve_out.data.tolist()[0]
				h = hem_out.data.tolist()[0]
				print(c,s,h)
				"""

				width,height = input.size
				points = [c[0]*width,c[1]*height,c[2]*width,c[3]*height,
						s[0]*width,s[1]*height,s[2]*width,s[3]*height,
						h[0]*width,h[1]*height,h[2]*width,h[3]*height]

				img = cv2.imread(RECV_PATH)

				for i, landmark in enumerate(zip(points[::2], points[1::2])):
					if landmark[0] > 0 and landmark[1] > 0:
						img = cv2.circle(img, (int(landmark[0]),int(landmark[1])), 5, (0,0,255), -1)

				cv2.imwrite(RECV_PATH, img)
				"""
				print("Drawing Done")

				#send_file(client)
				conn.send((c,s,h))
				conn.close()
				
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

	#params = filter(lambda p: p.requires_grad, model.parameters())
	criterion = nn.L1Loss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
#optimizer = torch.optim.Adagrad(params, lr = args.lr, weight_decay = args.weight_decay)

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)
		# train for one epoch
		train(image_data, model, criterion, optimizer, epoch, args.clothes)
		# evaluate on validation set
		dist = validate(val_data, model, args.clothes)
		#dist = 100
	    # remember best prec@1 and save checkpoint
		is_best = dist < best_dist
		best_dist = min(dist, best_dist)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'clothes_type':args.clothes,
			'state_dict': model.state_dict(),
			'best_distance': best_dist,
		}, is_best)

def validate(val_loader, model, clothes_type):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	distance = AverageMeter()
    
	model.eval()
	end = time.time()
	d = Drawing()
	if clothes_type == 0:
		for i, (input, collar, sleeve, hem, path) in enumerate(val_loader):
			data_time.update(time.time() - end)
			input_var = Variable(input, volatile=True)

			# compute output
			collar_out, sleeve_out, hem_out, _ = model(input_var)
			collar_list = collar_out.data.cpu().tolist()
			sleeve_list = sleeve_out.data.cpu().tolist()
			hem_list = hem_out.data.cpu().tolist()
			
			# drawing landmark point on images
			if args.draw:
				for img_path, c, s, h in zip(path,collar_list,sleeve_list,hem_list):
					img = Image.open(args.data + img_path)
					width,height = img.size
					d.draw_landmark_point(args.data + img_path, (0,0,255), 
																[c[0]*width,c[1]*height,c[2]*width,c[3]*height,
																s[0]*width,s[1]*height,s[2]*width,s[3]*height,
																h[0]*width,h[1]*height,h[2]*width,h[3]*height])

	        # Answers
			collar = Variable(collar,volatile =True).float().cuda(async=True)
			sleeve = Variable(sleeve,volatile =True).float().cuda(async=True)
			hem = Variable(hem,volatile =True).float().cuda(async=True)
			dist = accuracy((collar[:,2:6],sleeve[:,2:6],hem[:,2:6]),(collar_out,sleeve_out,hem_out))
			distance.update(dist.data[0], 1)
#print(path[0],sleeve[0],sleeve_out[0])
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
			input_var = Variable(input, volatile=True)

			waistline_out, hem_out, _ = model(input_var)
			waistline = Variable(waistline,volatile =True).float().cuda(async=True)
			hem = Variable(hem,volatile =True).float().cuda(async=True)
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
			input_var = Variable(input, volatile=True)

			# compute output
			collar_out, sleeve_out, waistline_out, hem_out, _ = model(input_var)

			# Answers
			collar = Variable(collar,volatile =True).float().cuda(async=True)
			sleeve = Variable(sleeve,volatile =True).float().cuda(async=True)
			waistline = Variable(waistline,volatile =True).float().cuda(async=True)
			hem = Variable(hem,volatile =True).float().cuda(async=True)

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

def train(train_loader, model, criterion, optimizer, epoch, clothes_type):
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
	end = time.time()
	if clothes_type == 0:
		for i, (input, collar, sleeve, hem, path) in enumerate(train_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			input_var = Variable(input)

			collar_out, sleeve_out, hem_out, _ = model(input_var)
			# Answers
			collar = Variable(collar).float().cuda()
			sleeve = Variable(sleeve).float().cuda()
			hem = Variable(hem).float().cuda()
			
			left_collar_indices = torch.nonzero((0 == ans_visuality[:,0].data))[:,0]
			right_collar_indices = torch.nonzero((0 == ans_visuality[:,1].data))[:,0]

			left_sleeve_indices = torch.nonzero((0 == ans_visuality[:,2].data))[:,0]
			right_sleeve_indices = torch.nonzero((0 == ans_visuality[:,3].data))[:,0]
			
			left_hem_indices = torch.nonzero((0 == ans_visuality[:,4].data))[:,0]
			right_hem_indices = torch.nonzero((0 == ans_visuality[:,5].data))[:,0]

			landmark = landmark.data
			ans_landmark = ans_landmark.data
			
			left_collar_loss = l1_criterion(Variable(torch.index_select(landmark[:,0:2],0,left_collar_indices)), Variable(torch.index_select(ans_landmark[:,0:2],0,left_collar_indices)))
			right_collar_loss = l1_criterion(Variable(torch.index_select(landmark[:,2:4],0,right_collar_indices)), Variable(torch.index_select(ans_landmark[:,2:4],0,right_collar_indices)))

			left_sleeve_loss = l1_criterion(Variable(torch.index_select(landmark[:,4:6],0,left_sleeve_indices)), Variable(torch.index_select(ans_landmark[:,4:6],0,left_sleeve_indices)))
			right_sleeve_loss = l1_criterion(Variable(torch.index_select(landmark[:,6:8],0,right_sleeve_indices)), Variable(torch.index_select(ans_landmark[:,6:8],0,right_sleeve_indices)))

			left_hem_loss = l1_criterion(Variable(torch.index_select(landmark[:,12:14],0,left_hem_indices)), Variable(torch.index_select(ans_landmark[:,8:10],0,left_hem_indices)))
			right_hem_loss = l1_criterion(Variable(torch.index_select(landmark[:,14:16],0,right_hem_indices)), Variable(torch.index_select(ans_landmark[:,10:12],0,right_hem_indices)))


			hem_loss = criterion(hem_out, hem[:,2:6])
			collar_loss =criterion(collar_out, collar[:,2:6])
			sleeve_loss = criterion(sleeve_out, sleeve[:,2:6])

			loss = hem_loss + collar_loss + sleeve_loss

			dist = accuracy((collar[:,2:6],sleeve[:,2:6],hem[:,2:6]),(collar_out,sleeve_out,hem_out))
			losses.update(loss.data[0], input.size(0))
			distance.update(dist.data[0], 1)

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
						'Location Loss {loss.val:.4f} ({loss.avg:.4f})\t'
						'Distances {dist.val:.4f} ({dist.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time, loss = losses, dist = distance))

	elif clothes_type == 1:
		for i, (input, waistline, hem, path) in enumerate(train_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			input_var = Variable(input)
			optimizer.zero_grad()
			waistline_out, hem_out, _ = model(input_var)

			# Answers
			waistline = Variable(waistline,volatile =True).float().cuda(async=True)
			hem = Variable(hem,volatile =True).float().cuda(async=True)

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
			collar = Variable(collar,volatile =True).float().cuda(async=True)
			sleeve = Variable(sleeve,volatile =True).float().cuda(async=True)
			waistline = Variable(waistline,volatile =True).float().cuda(async=True)
			hem = Variable(hem,volatile =True).float().cuda(async=True)

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
	lr = args.lr * (0.1 ** (epoch // 20))
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
	client.close()

	return l
		
def send_file(client):
	with open(RECV_PATH, 'rb') as f:
		l = f.read(BUFFER_SIZE)

		while (l):
			client.send(l)
			l = f.read(BUFFER_SIZE)

	client.shutdown(socket.SHUT_WR)
	print("Done Sending")


if __name__ == '__main__':
	main()    
