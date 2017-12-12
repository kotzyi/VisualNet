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
#from models.LandmarkNet_Land import landmarknet_land
#from models.LandmarkNet_Vis import landmarknet_vis
#from models.LandmarkNet import landmarknet
from models.Tuned_DRN import  drn_d_38
import torchvision.models as models
import random
from utils.drawing import Drawing
from PIL import Image
import socket
import struct
import cv2
import numpy as np
import pickle
import math
import pandas as pd
import utils.networks as NM
import utils.communicate as comm
import utils.data as DM

TCP_IP = '10.214.35.36'
TCP_PORT = 5005
BUFFER_SIZE = 4096
IMG_SIZE = 320
RECV_PATH = '/home/jlee/VisualNet/received_img.jpg'
WINDOW_SIZE = 4

attr_class = pd.read_table("/data/deep-fashion/Anno/list_category_cloth.txt",sep=r"  +",engine='python')
#attr_class = attr_class['attribute_name'].values.tolist()
attr_class = attr_class['category_name'].values.tolist()
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
parser.add_argument('-l','--land', nargs='+',
                    help='model name, file_name, and path to latest checkpoint (ex: land model name, file name, architecture, land model save path)')
parser.add_argument('-v','--visuality', nargs='+',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--attribute', nargs='+',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--feature-resume', default='', type=str,
					help='path to latest checkpoint (default: none)')
parser.add_argument('--attr-resume', default='', type=str,
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
parser.add_argument('--save-file', default='', type=str,
					help='save file for evaluation results')

def main():
	#기본 설정 부분
	global args, best_acc
	args = parser.parse_args()
	data_path = args.data
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	pin = args.cuda

	best_dist = 100.0

	if args.cuda:
		print("GPU Mode")
	else:
		print("CPU Mode")

	datas = DM.Data(args.batch_size, IMG_SIZE = IMG_SIZE, workers = args.workers, CUDA = args.cuda)

	lm_eval_name = 'lm_eval_data'
	lm_data_module = 'landmark_loader'
	lm_params = (data_path, False, args.clothes)
	datas.load(lm_eval_name, lm_data_module, True, *lm_params)

	if args.evaluate:
		validate(datas.get(lm_data_name), model)
		return

	lm_train_name = 'lm_train_data'
	lm_data_module = 'landmark_loader'
	lm_params = (data_path, True, args.clothes)
	datas.load(lm_train_name, lm_data_module, True, *lm_params)

	#params = filter(lambda p: p.requires_grad, model.parameters())
	criterion = nn.L1Loss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
	#optimizer = torch.optim.Adagrad(params, lr = args.lr, weight_decay = args.weight_decay)

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		train(datas.get(lm_train_name), model, criterion, optimizer, epoch)

		# evaluate on validation set
		dist = validate(datas.get(lm_eval_name), model)

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

def validate(val_loader, model):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
   
	lc_distance = AverageMeter()
	rc_distance = AverageMeter()
	ls_distance = AverageMeter()
	rs_distance = AverageMeter()
	lh_distance = AverageMeter()
	rh_distance = AverageMeter()
	all_distance =AverageMeter()
	model.eval()
	end = time.time()
	d = Drawing()

	if clothes_type == 0:
		for i, (input, collar, sleeve, hem, path) in enumerate(val_loader):
			data_time.update(time.time() - end)
			input_var = Variable(input, volatile=True)

			# compute output
			landmark = model(input_var)
			landmark = landmark.data
            # Answers
			ans_landmark = torch.cat((collar[:,2:6],sleeve[:,2:6],hem[:,2:6]),1).float().cuda()
			ans_visuality = torch.cat((collar[:,0:2],sleeve[:,0:2],hem[:,0:2]),1).long().cuda()

			left_collar_indices = torch.nonzero((0 == ans_visuality[:,0]))[:,0].cuda()
			right_collar_indices = torch.nonzero((0 == ans_visuality[:,1]))[:,0].cuda()

			left_sleeve_indices = torch.nonzero((0 == ans_visuality[:,2]))[:,0].cuda()
			right_sleeve_indices = torch.nonzero((0 == ans_visuality[:,3]))[:,0].cuda()
			
			left_hem_indices = torch.nonzero((0 == ans_visuality[:,4]))[:,0].cuda()
			right_hem_indices = torch.nonzero((0 == ans_visuality[:,5]))[:,0].cuda()

			lc_dist = accuracy(landmark[:,0:2].index_select(0,left_collar_indices),ans_landmark[:,0:2].index_select(0,left_collar_indices))
			rc_dist = accuracy(landmark[:,2:4].index_select(0,right_collar_indices),ans_landmark[:,2:4].index_select(0,right_collar_indices))
			ls_dist = accuracy(landmark[:,4:6].index_select(0,left_sleeve_indices),ans_landmark[:,4:6].index_select(0,left_sleeve_indices))
			rs_dist = accuracy(landmark[:,6:8].index_select(0,right_sleeve_indices),ans_landmark[:,6:8].index_select(0,right_sleeve_indices))
			lh_dist = accuracy(landmark[:,12:14].index_select(0,left_hem_indices),ans_landmark[:,8:10].index_select(0,left_hem_indices))
			rh_dist = accuracy(landmark[:,14:16].index_select(0,right_hem_indices),ans_landmark[:,10:12].index_select(0,right_hem_indices))

			lc_distance.update(lc_dist, 1)
			rc_distance.update(rc_dist, 1)
			ls_distance.update(ls_dist, 1)
			rs_distance.update(rs_dist, 1)
			lh_distance.update(lh_dist, 1)
			rh_distance.update(rh_dist, 1)
			all_distance.update(lc_dist+rc_dist+ls_dist+rs_dist+lh_dist+rh_dist,1)

			# drawing landmark point on images
			if args.draw:
				for img_path, c, s, h in zip(path,collar_list,sleeve_list,hem_list):
					img = Image.open(args.data + img_path)
					width,height = img.size
					d.draw_landmark_point(args.data + img_path, (0,0,255), 
																[c[0]*width,c[1]*height,c[2]*width,c[3]*height,
																s[0]*width,s[1]*height,s[2]*width,s[3]*height,
																h[0]*width,h[1]*height,h[2]*width,h[3]*height])

	        # Answer
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Epoch: [{0}/{1}]\t'
						'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
						'LC_Dists {lc.val:.3f} ({lc.avg:.3f})\t'
                        'RC_Dists {rc.val:.3f} ({rc.avg:.3f})\t'
                        'LS_Dists {ls.val:.3f} ({ls.avg:.3f})\t'
                        'RS_Dists {rs.val:.3f} ({rs.avg:.3f})\t'
                        'LH_Dists {lh.val:.3f} ({lh.avg:.3f})\t'
                        'RH_Dists {rh.val:.3f} ({rh.avg:.3f})\t'
						'Distance {all.val:.3f} ({all.avg:.3f})'
						.format(i, len(val_loader), batch_time=batch_time,data_time=data_time,  lc = lc_distance, rc = rc_distance, ls = ls_distance, rs = rs_distance, lh = lh_distance, rh = rh_distance, all = all_distance))

	
	print('Test Result: Distances ({all.avg:.4f})'.format(all = all_distance))
	return all_distance.avg

def train(train_loader, model, criterion, optimizer, epoch):
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
	lc_distance = AverageMeter()
	rc_distance = AverageMeter()
	ls_distance = AverageMeter()
	rs_distance = AverageMeter()
	lh_distance = AverageMeter()
	rh_distance = AverageMeter()

	# switch to train mode
	model.train()
	end = time.time()
	if clothes_type == 0:
		for i, (input, collar, sleeve, hem, path) in enumerate(train_loader):
			# measure data loading time
			data_time.update(time.time() - end)
			input_var = Variable(input)

			landmark = model(input_var)

			# Answers
			ans_landmark = torch.cat((collar[:,2:6],sleeve[:,2:6],hem[:,2:6]),1)
			ans_visuality = Variable(torch.cat((collar[:,0:2],sleeve[:,0:2],hem[:,0:2]),1)).long().cuda(async=True)

			ans_landmark = Variable(ans_landmark).float().cuda(async=True)


			left_collar_indices = Variable(torch.nonzero((0 == ans_visuality[:,0].data))[:,0])
			right_collar_indices = Variable(torch.nonzero((0 == ans_visuality[:,1].data))[:,0])

			left_sleeve_indices = Variable(torch.nonzero((0 == ans_visuality[:,2].data))[:,0])
			right_sleeve_indices = Variable(torch.nonzero((0 == ans_visuality[:,3].data))[:,0])
			
			left_hem_indices = Variable(torch.nonzero((0 == ans_visuality[:,4].data))[:,0])
			right_hem_indices = Variable(torch.nonzero((0 == ans_visuality[:,5].data))[:,0])

			#landmark = landmark.data
			#ans_landmark = ans_landmark.data
			left_collar_loss = criterion(landmark[:,0:2].index_select(0,left_collar_indices), ans_landmark[:,0:2].index_select(0,left_collar_indices))
			right_collar_loss = criterion(landmark[:,2:4].index_select(0,right_collar_indices), ans_landmark[:,2:4].index_select(0,right_collar_indices))

			left_sleeve_loss = criterion(landmark[:,4:6].index_select(0,left_sleeve_indices), ans_landmark[:,4:6].index_select(0,left_sleeve_indices))
			right_sleeve_loss = criterion(landmark[:,6:8].index_select(0,right_sleeve_indices), ans_landmark[:,6:8].index_select(0,right_sleeve_indices))

			left_hem_loss = criterion(landmark[:,12:14].index_select(0,left_hem_indices), ans_landmark[:,8:10].index_select(0,left_hem_indices))
			right_hem_loss = criterion(landmark[:,14:16].index_select(0,right_hem_indices), ans_landmark[:,10:12].index_select(0,right_hem_indices))


			loss = left_hem_loss + right_hem_loss + left_collar_loss + right_collar_loss + left_sleeve_loss + right_sleeve_loss
			losses.update(loss.data[0], input.size(0))

			lc_dist = accuracy(landmark[:,0:2].index_select(0,left_collar_indices),ans_landmark[:,0:2].index_select(0,left_collar_indices))
			rc_dist = accuracy(landmark[:,2:4].index_select(0,right_collar_indices),ans_landmark[:,2:4].index_select(0,right_collar_indices))
			ls_dist = accuracy(landmark[:,4:6].index_select(0,left_sleeve_indices),ans_landmark[:,4:6].index_select(0,left_sleeve_indices))
			rs_dist = accuracy(landmark[:,6:8].index_select(0,right_sleeve_indices),ans_landmark[:,6:8].index_select(0,right_sleeve_indices))
			lh_dist = accuracy(landmark[:,12:14].index_select(0,left_hem_indices),ans_landmark[:,8:10].index_select(0,left_hem_indices))
			rh_dist = accuracy(landmark[:,14:16].index_select(0,right_hem_indices),ans_landmark[:,10:12].index_select(0,right_hem_indices))

			lc_distance.update(lc_dist.data[0], 1)
			rc_distance.update(rc_dist.data[0], 1)
			ls_distance.update(ls_dist.data[0], 1)
			rs_distance.update(rs_dist.data[0], 1)
			lh_distance.update(lh_dist.data[0], 1)
			rh_distance.update(rh_dist.data[0], 1)

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
						'LC_Dists {lc.val:.3f} ({lc.avg:.3f})\t'
						'RC_Dists {rc.val:.3f} ({rc.avg:.3f})\t'
						'LS_Dists {ls.val:.3f} ({ls.avg:.3f})\t'
						'RS_Dists {rs.val:.3f} ({rs.avg:.3f})\t'
						'LH_Dists {lh.val:.3f} ({lh.avg:.3f})\t'
						'RH_Dists {rh.val:.3f} ({rh.avg:.3f})\t'
						.format(epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time, loss = losses, lc = lc_distance, rc = rc_distance, ls = ls_distance, rs = rs_distance, lh = lh_distance, rh = rh_distance))


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
	
	distances = dist_function(output, target)
	
	return distances.mean() / 2

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

def get_feature_vector(coords,feature):
	"""
	1. Take off patches from the feature map at 8 landmark points
	2. Stack the patches. ex) n x 512 x 32 x 32 -> n x 4096x 3 x 3
	3. Find Maximum Activation of the patches n x 4096
	n = batch size
	c = channel
	m = map size
	feature_vecs =  feature vectors of images that represent the images

	feature = feature map from landmark model
	coords = coordination of landmarks
	"""
	feature_vecs = []
	n, c, m, _ = feature.size()
	w = int(WINDOW_SIZE / 2)

	# n x c x m x m
	pad = nn.ReflectionPad2d(w)
	feature = pad(feature).data

	# n x c x (m + pad) x (m + pad)
	coords = [coords]
	for i, coord in enumerate(coords):
		patches = []
		for x,y in zip(coord[0::2],coord[1::2]):
			if x >= 0 and x < 1 and y < 1:
				x = math.floor(x * m)
				y = math.floor(y * m)
				patch = feature[i, :, x : x + WINDOW_SIZE , y : y + WINDOW_SIZE].contiguous()
				_, a, b = patch.size()
				patch_vec, _ = torch.max(patch.view(c, WINDOW_SIZE * WINDOW_SIZE), 1) # MAC
				#patch_vec = patch.view(WINDOW_SIZE * WINDOW_SIZE * c) # COS
				patches = patches + patch_vec.tolist()

			else:
				patches = patches + [0] * c # MAC
				#patches = patches + [0] * (WINDOW_SIZE * WINDOW_SIZE * c) # COS

		feature_vecs.append(patches)

	return feature_vecs

def pickle_load(filename):
	data = []
	with open(filename,'rb') as fp:
		while True:
			try:
				p, d = pickle.load(fp)
				data.append((p,d))
			except EOFError:
				break

	return data

if __name__ == '__main__':
	main()    
