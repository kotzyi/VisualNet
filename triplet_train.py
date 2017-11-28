'''To-Do
1. 학습시에 모든 attribute를 사용하는 형태로 학습 코드 변경할 것!
2. category and attribute prediction data 사용할 것!
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
from data.image_loader import AnchorFolder
from data.eval_loader import EvalFolder
from models.LandmarkNet import landmarknet
from models.VisualNet import visualnet
import math
import pickle
import sklearn.metrics.pairwise
import scipy.spatial.distance

IMG_SIZE = 320
WINDOW_SIZE = 3

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
                    help='path to latest checkpoint of landmarknet(default: none)')
parser.add_argument('--vis-load', default='', type=str,
					help='path to latest checkpoint of visualnet(default: none)')
parser.add_argument('--name', default='Visual_Search', type=str,
                    help='name of experiment')
parser.add_argument('--workers', type = int, default = 8, metavar = 'N',
					help='number of works for data londing')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,	metavar='W', 
					help='weight decay (default: 1e-4)')
parser.add_argument('--anchor', default='', type=str,
					help='path to anchor image folder')
parser.add_argument('--feature-size', default=1000, type=int,
					help='fully connected layer size')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('-a', '--arch', default='resnet50', type=str,
					help='Architecture of network for training in resnet18,34,50,101,152')
parser.add_argument('-c', '--clothes', default=0, type=int,
					help='Clothes type:0 - upper body, 1 - lower body, 2 - full body')
parser.add_argument('--save-file', default='', type=str,
					help='save file for evaluation results')


best_acc = 0

def main():
	#기본 설정 부분
	global args, best_acc, pin
	args = parser.parse_args()
	data_path = args.data
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	pin = True
	if args.cuda:
		print("GPU Mode")
	else:
		pin = False
		print("CPU Mode")
	
	land_model = landmarknet(args.arch,args.clothes)
#visual_model = visualnet(os.path.isfile(args.vis_load), args.arch,args.clothes)

	if args.cuda:
		land_model = torch.nn.DataParallel(land_model).cuda()
#visual_model = torch.nn.DataParallel(visual_model).cuda()
#		triplet_loss = nn.TripletMarginLoss(margin=20.0, p=2).cuda()

	print("Model parameters are loaded")
	
	# optionally resume from a checkpoint
	if args.land_load:
		if os.path.isfile(args.land_load):
			print("=> loading checkpoint '{}'".format(args.land_load))
			checkpoint = torch.load(args.land_load)
			args.start_epoch = checkpoint['epoch']
			args.clothes = checkpoint['clothes_type']
			land_model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})".format(args.land_load, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.land_load))

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	"""
	if args.vis_load:
		if os.path.isfile(args.vis_load):
			print("=> loading checkpoint '{}'".format(args.vis_load))
			checkpoint = torch.load(args.vis_load)
			args.start_epoch = checkpoint['epoch']
			visual_model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})".format(args.vis_load, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.vis_load))

	"""
	if args.evaluate:
		if os.path.isfile(args.save_file):
			anchor_image = torch.utils.data.DataLoader(
				EvalFolder(args.data,transforms.Compose([
					transforms.Scale(IMG_SIZE),
					transforms.CenterCrop(IMG_SIZE),
					transforms.ToTensor(),
					normalize,
				])),
				batch_size = args.batch_size,
				shuffle = False,
				num_workers = args.workers,
				pin_memory = pin,
			)
			print("Complete Data loading(%s)" % len(anchor_image))
			validate(anchor_image, land_model)#, visual_model)

		else:
			val_data = torch.utils.data.DataLoader(
				EvalFolder(data_path,transforms.Compose([
					transforms.Scale(IMG_SIZE),
					transforms.CenterCrop(IMG_SIZE),
					transforms.ToTensor(),
					normalize,
				])),
				batch_size=args.batch_size,
				shuffle=False,
				num_workers = args.workers,
				pin_memory = pin,
			)
			print("Complete Data loading(%s)" % len(val_data))

			validate(val_data, land_model)#, visual_model)

		return

	#triplet data loading
	"""
	triplet_train_data = torch.utils.data.DataLoader(
		TripletFolder(data_path,transforms.Compose([
			transforms.Scale(256),
			transforms.CenterCrop(256),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=args.batch_size, 
		shuffle=True,
		num_workers = args.workers,
		pin_memory = pin,
	)
	"""
	print("Complete Data loading(%s)" % len(triplet_train_data))

	params = filter(lambda p: p.requires_grad, visual_model.parameters())
	optimizer = torch.optim.Adagrad(params, args.lr,weight_decay=args.weight_decay)


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

def validate(val_data, land_model):#, visual_model):
	land_model.eval()

	if os.path.isfile(args.save_file):
		cos = nn.CosineSimilarity(dim=2)
		results = pickle_load(args.save_file)
		#waistline_out = Variable(torch.Tensor(args.batch_size,4).fill_(-0.1),volatile = True).cuda()
		#visual_model.eval()
		while True:
			anchor_path = input("PATH: ")
			if anchor_path == "q":
				break
			else:
				for i, (path, image, landmarks) in enumerate(val_data):				
					#for image in anchor_image:
					input_var = Variable(image,volatile = True)
					centroid = torch.DoubleTensor(get_centroid(landmarks))
					landmarks[:,8:10] = centroid
					_,_,_, feature = land_model(input_var)

					feature_vec = get_feature_vector(landmarks, feature)
					feature_var = Variable(torch.Tensor(feature_vec).cuda(),volatile = True) # for 1036_train.pickle

#isLandmark_inf = exist_landmark(feature_var[0]);
#splited_inference = torch.split(feature_var,64, dim=1)
					scores = []

					for pathes, datas in results:
						data_var = Variable(torch.Tensor(datas).cuda(),volatile=True)
						r,c = data_var.size()
						data_var = data_var.view(-1,8, int(c/8))
						mask_data = data_var.ne(0).float()

						splited_feature_var = feature_var.expand(r,c).contiguous().view(-1, 8, int(c/8))
						mask_feature = splited_feature_var.ne(0).float()
						
						data_var = data_var * mask_feature
						splited_feature_var = splited_feature_var * mask_data

						similarities = cos(data_var, splited_feature_var)
						zeros = torch.sum(similarities.gt(0).float(),1)	

						max_score, max_index = torch.max(torch.sum(similarities,1),0)
						scores.append((pathes[max_index.data[0]],max_score.data[0],zeros[max_index.data[0]].data[0]))


					
					top3 = sorted(scores, key = lambda x:x[1])
					print(top3[-10:])
					"""
						for (path, data) in zip(pathes, datas):
							l1 = []
							l2 = []
							data = Variable(torch.unsqueeze(torch.Tensor(data),0),volatile = True).cuda()
							splited_data = torch.split(data,64,dim=1)
							isLandmark_data = exist_landmark(data[0])
							score = 0
							for j in range(8):
								if isLandmark_inf[j] + isLandmark_data[j] == 0:
									l1.append(splited_data[j])
									l2.append(splited_inference[j])

							if len(l2) > 2:
								score += cos(torch.cat(l1,dim=1),torch.cat(l2,dim=1)).data.tolist()[0] - (7 - len(l1)) * 0.005
								scores.append((path,score,isLandmark_data))
							score = cos(data,feature_var).data.tolist()[0]
							scores.append((path,score,isLandmark_data))

					top3 = sorted(scores, key = lambda x:x[1])
					print(top3[-10:])
					"""
	
	else:
		#rogress bar setting
		counter = 0
		imageN = len(val_data)*args.batch_size
		printProgressBar(counter, imageN, prefix = 'Progress:', suffix = 'Complete', length = 100)

		with open(args.save_file,"wb") as fp:
			for i, (path, image, landmarks) in enumerate(val_data):
				input_var = Variable(image,volatile = True)
				#centroid = torch.DoubleTensor(get_centroid(landmarks))
				#landmarks[:,8:10] = centroid

				_,_,_, feature = land_model(input_var)

				feature_vec = get_feature_vector(landmarks, feature)

				#for LANDMARK DETECTOR
				pickle.dump((path,feature_vec),fp,protocol=pickle.HIGHEST_PROTOCOL)
				counter = counter + args.batch_size
				printProgressBar(counter, imageN, prefix = 'Progress:', suffix = 'Complete', length = 100)	

def train(epoch, triplet_data, land_model, visual_model, triplet_loss, optimizer):
	losses = AverageMeter()
	pos_distance = AverageMeter()
	neg_distance = AverageMeter()
	pdist = nn.PairwiseDistance(p=2)

	land_model.eval()
	visual_model.train()
	waistline_out = Variable(torch.Tensor(args.batch_size,4).fill_(-0.1)).cuda()
	for i, (anchor, anc_ans, positive, pos_ans, negative, neg_ans) in enumerate(triplet_data):
		optimizer.zero_grad()

		input_var = Variable(anchor)
		collar_out, sleeve_out, hem_out, anchor_feature = land_model(input_var)
#		anchor_pool5_layer = gen_pool5_layer((collar_out[:,0:2], collar_out[:,2:4], sleeve_out[:,0:2], sleeve_out[:,2:4],
#				 waistline_out[:,0:2], waistline_out[:,2:4], hem_out[:,0:2], hem_out[:,2:4]), anchor_feature)

		ans = Variable(anc_ans).float().cuda(async=True)
		pool5_layer = gen_pool5_layer((ans[:,2:4],ans[:,4:6],ans[:,8:10],ans[:,10:12],waistline_out[:,0:2], waistline_out[:,2:4],ans[:,14:16],ans[:,16:18]), anchor_feature)
		MAC = gen_MAC(pool5_layer)
		embedded_anchor = visual_model(anchor_feature, pool5_layer)

		input_var = Variable(positive)
		collar_out, sleeve_out, hem_out, positive_feature = land_model(input_var)
#positive_pool5_layer = gen_pool5_layer((collar_out[:,0:2], collar_out[:,2:4], sleeve_out[:,0:2], sleeve_out[:,2:4],
#				 waistline_out[:,0:2], waistline_out[:,2:4], hem_out[:,0:2], hem_out[:,2:4]), positive_feature)
		ans = Variable(pos_ans).float().cuda(async=True)
		pool5_layer = gen_pool5_layer((ans[:,2:4],ans[:,4:6],ans[:,8:10],ans[:,10:12],waistline_out[:,0:2], waistline_out[:,2:4],ans[:,14:16],ans[:,16:18]), positive_feature)
		embedded_positive = visual_model(positive_feature, pool5_layer)

		input_var = Variable(negative)
		collar_out, sleeve_out, hem_out, negative_feature = land_model(input_var)
#		negative_pool5_layer = gen_pool5_layer((collar_out[:,0:2], collar_out[:,2:4], sleeve_out[:,0:2], sleeve_out[:,2:4],
#				 waistline_out[:,0:2], waistline_out[:,2:4], hem_out[:,0:2], hem_out[:,2:4]), negative_feature)

		ans = Variable(neg_ans).float().cuda(async=True)
		pool5_layer = gen_pool5_layer((ans[:,2:4],ans[:,4:6],ans[:,8:10],ans[:,10:12],waistline_out[:,0:2], waistline_out[:,2:4],ans[:,14:16],ans[:,16:18]), negative_feature)
		embedded_negative = visual_model(negative_feature, pool5_layer)	

		loss = triplet_loss(embedded_anchor, embedded_positive, embedded_negative)
		
		loss.backward()
		optimizer.step()
	
		losses.update(loss.data[0], anchor.size(0))
		pos_distance.update(torch.sum(pdist(embedded_anchor,embedded_positive)).data.cpu().tolist()[0]/args.batch_size)
		neg_distance.update(torch.sum(pdist(embedded_anchor,embedded_negative)).data.cpu().tolist()[0]/args.batch_size)

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'	
					'Triplet Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Positive Distance {pos_dist.val:.4f} ({pos_dist.avg:.4f})'
					'Negative Distance {neg_dist.val:.4f} ({neg_dist.avg:.4f})'.format(epoch, i, len(triplet_data), loss=losses, pos_dist = pos_distance, neg_dist = neg_distance))

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

def exist_landmark(t):
	isLandmark = []
	for j in range(8):
		if torch.mean(t.data[(j*64):(j*64+10)]) != 0:
			isLandmark.append(0)
		else:
			isLandmark.append(1)

	return isLandmark

def get_centroid(landmarks):
	centroids = []

	for landmark in landmarks:
		centroid = [0,0]
		n_points = 0

		for x,y in zip(landmark[0::2], landmark[1::2]):
			if x != -0.1:
				centroid[0] += x
				centroid[1] += y
				n_points += 1
		if n_points > 0:
			centroid[0] = centroid[0] / n_points
			centroid[1] = centroid[1] / n_points
		
		centroids.append(centroid)

	return centroids

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
	for i, coord in enumerate(coords):
		patches = []
		for x,y in zip(coord[0::2],coord[1::2]):
			if x >= 0 and x < 1 and y < 1:
				x = math.floor(x * m)
				y = math.floor(y * m)
				patch = feature[i, :, x : x + WINDOW_SIZE , y : y + WINDOW_SIZE].contiguous()
				_, a, b = patch.size()
				#patch_vec, _ = torch.max(patch.view(c, WINDOW_SIZE * WINDOW_SIZE), 1) # MAC
				patch_vec = patch.view(WINDOW_SIZE * WINDOW_SIZE * c) # COS
				patches = patches + patch_vec.tolist()

			else:
				#patches = patches + [0] * c # MAC
				patches = patches + [0] * (WINDOW_SIZE * WINDOW_SIZE * c) # COS

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
