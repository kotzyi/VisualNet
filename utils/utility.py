import torch
import torch.nn as nn
import pickle
import math
import shutil
import pandas as pd
from torchvision import transforms


cat_table = pd.read_table("/data/deep-fashion/Anno/list_category_cloth.txt",sep=r"  +",engine='python')
cat_class = cat_table['category_name'].values.tolist()
attr_table = pd.read_table("/data/deep-fashion/Anno/list_attr_cloth.txt",sep=r"  +",engine='python')
attr_class = attr_table['attribute_name'].values.tolist()

cat_weight = [5,4,1,5,5,3,5,5,5,5,
              4,5,5,5,5,3,3,1,4,5,
              5,5,5,5,5,4,5,5,5,5,
              5,2,3,5,5,5,5,5,5,5,
              1,4,5,5,5,5,5,4,5,5]
att_weight = [0.833, 1.01, 0.943, 2.083, 2.128, 0.98, 0.61, 0.498, 0.962, 1.961, 0.68, 2.041, 0.699, 0.917, 0.417, 0.61, 1.25, 0.524, 0.552, 2.041, 0.613, 0.467, 0.725, 0.694, 0.575, 0.437, 0.629, 1.266, 0.521, 0.602, 2.041, 2.273, 0.833, 1.493, 0.775, 1.786, 0.552, 0.543, 0.694, 0.415, 0.394, 0.442, 0.546, 0.538, 5.556, 0.498, 0.943, 2.174, 0.699, 0.478, 0.495, 0.73, 1.961, 1.923, 1.538, 0.552, 1.389, 0.99, 5.882, 1.351, 1.667, 0.685, 1.613, 1.351, 2.128, 1.351, 0.833, 1.887, 0.645, 0.621, 0.394, 0.758, 0.417, 0.407, 0, 2.326, 4.545, 0.84, 0.787, 1.333, 0.862, 0.463, 2.273, 0.417, 0.578, 0.413, 1.25, 0.971, 2.083, 1.887, 1.471, 16.666, 2.174, 10.0, 0.625, 1.493, 2.273, 1.493, 1.176, 0.769, 1.429, 0.735, 1.282, 3.448, 0.459, 2.703, 0.73, 1.25, 0.407, 0.877, 2.632, 24.999, 1.0, 16.666, 12.5, 0.532, 4.348, 0.408, 1.852, 0.417, 1.25, 0.704, 1.667, 0.658, 0.654, 0.408, 2.5, 2.174, 0.392, 2.564, 2.326, 0.493, 0.417, 2.326, 2.273, 0.735, 0.559, 0.68, 1.471, 0.437, 1.695, 0.442, 0.602, 1.087, 0.952, 1.667, 2.857, 1.754, 0.758, 0.806, 0.505, 1.149, 0.667, 1.19, 4.167, 1.562, 1.25, 0.429, 1.471, 8.333, 0.69, 1.923, 1.0, 0.98, 1.316, 1.389, 0.461, 0.546, 0.578, 1.515, 1.22, 0.99, 1.266, 0.862, 1.176, 0.847, 8.333, 0.633, 0.408, 0.637, 0.427, 0.654, 1.724, 1.149, 0.463, 1.01, 0.763, 1.408, 0.617, 0.395, 2.128, 1.639, 1.351, 2.273, 0.82, 0.699, 0.4, 1.754, 1.0, 0.637, 0.709, 0.51, 1.587, 6.25, 3.03, 1.163, 0.556, 0.565, 0.467, 0.455, 0.552, 1.316, 1.818, 1.205, 2.128, 0.901, 1.02, 0.694, 0, 0.524, 1.429, 1.538, 0.735, 7.692, 1.235, 1.299, 2.857, 14.286, 2.222, 1.818, 0.82, 0.73, 20.0, 1.515, 1.538, 0.403, 0.392, 1.351, 0.926, 1.852, 2.083, 0.917, 1.176, 0.521, 0.503, 2.174, 0.402, 0.606, 1.064, 0.395, 0.51, 1.111, 2.381, 2.083, 0.559, 1.754, 0.621, 1.695, 5.0, 0.935, 1.887, 0.592, 0.478, 1.667, 0.658, 0.82, 1.562, 1.724, 0.575, 2.128, 1.136, 0.505, 0.581, 0, 0.437, 2.174, 0.735, 11.111, 1.075, 1.136, 0.735, 1.562, 1.316, 1.316, 0.559, 0.526, 1.493, 1.064, 1.961, 1.754, 1.852, 0.671, 0.446, 0.559, 2.273, 1.471, 2.083, 1.695, 1.149, 0.49, 20.0, 2.222, 0.73, 0.68, 1.562, 0.769, 0.654, 0.405, 33.332, 10.0, 1.124, 1.695, 0.446, 0.448, 2.041, 1.053, 1.818, 0.395, 3.03, 1.02, 1.923, 0.662, 0.877, 1.408, 0.505, 0.694, 1.724, 0.493, 1.449, 1.695, 0.935, 0.469, 0.649, 0.901, 2.857, 1.299, 5.0, 0.495, 1.111, 0.637, 2.041, 1.149, 1.064, 2.273, 2.174, 2.222, 0.441, 0.437, 1.22, 0.398, 1.333, 0.746, 0.549, 3.846, 0.606, 2.0, 1.818, 2.083, 0.714, 0.402, 1.282, 0.769, 0.813, 1.37, 0.503, 0.833, 2.273, 1.587, 1.923, 1.818, 1.667, 1.695, 1.562, 1.205, 1.613, 1.724, 1.111, 0.671, 1.961, 1.299, 0.667, 0.685, 2.128, 1.25, 0.943, 5.263, 0.893, 2.128, 0.61, 0.392, 1.471, 0.446, 0.725, 0.461, 1.333, 0.588, 1.639, 0.645, 0.893, 0.69, 0.629, 0.476, 0.613, 1.724, 1.613, 3.846, 1.031, 0.781, 0.535, 1.587, 1.408, 0.578, 1.695, 0.885, 0.629, 0.69, 2.083, 1.01, 0.392, 1.316, 1.205, 0.488, 2.439, 0.645, 1.087, 1.515, 1.333, 2.273, 1.111, 0.714, 1.149, 0.719, 1.471, 0.833, 6.667, 0.775, 1.075, 1.064, 0.926, 1.724, 0.441, 0.781, 2.564, 2.381, 1.266, 0.602, 3.846, 0.709, 1.818, 7.692, 0.415, 0.549, 1.613, 0.418, 0.505, 2.083, 1.562, 7.143, 0.769, 1.639, 0.935, 1.429, 0.741, 0.562, 2.273, 24.999, 2.222, 7.692, 0.602, 0.741, 0.571, 0.917, 0.909, 3.846, 1.923, 0.392, 0.437, 2.439, 1.316, 1.961, 0.694, 0.617, 0.549, 0.758, 1.351, 0.398, 0.935, 0.459, 1.471, 1.562, 0.926, 2.0, 0.529, 0.758, 0.602, 4.167, 2.439, 1.408, 0.444, 1.282, 0.606, 0.51, 1.37, 1.408, 1.754, 2.0, 2.128, 0.99, 1.667, 0.98, 2.941, 0.98, 0.478, 3.125, 1.961, 3.03, 1.266, 1.19, 0.493, 0.787, 0.532, 1.075, 0.532, 0.935, 2.041, 0.562, 1.205, 2.381, 1.493, 0.498, 0.99, 3.333, 5.556, 0.649, 0.495, 24.999, 0.877, 1.667, 0.662, 0.826, 0.571, 0.588, 0.746, 7.143, 0.926, 1.0, 0.51, 0.806, 0.412, 1.695, 0.709, 2.083, 1.961, 0.84, 1.887, 2.0, 1.818, 1.538, 1.176, 0.971, 0.415, 0.833, 16.666, 2.857, 2.222, 0.439, 1.235, 0.513, 0.741, 0.408, 0.552, 1.449, 1.351, 0.926, 0.526, 1.515, 3.226, 3.125, 0.935, 0.444, 1.235, 1.923, 2.174, 1.235, 1.786, 5.882, 0.599, 0.971, 1.01, 0.559, 1.02, 0.49, 0.599, 0.526, 1.818, 1.562, 0.585, 0.806, 0.448, 11.111, 1.724, 0.568, 0.893, 0.855, 0.524, 1.786, 0.862, 1.786, 0.581, 0.746, 6.667, 0.483, 1.075, 1.111, 1.02, 0.794, 1.923, 1.754, 2.0, 0.559, 0.513, 0.943, 1.695, 1.19, 0.617, 0.493, 0.862, 0.518, 0.51, 1.205, 1.754, 1.429, 1.852, 0.529, 0.405, 0.806, 0.575, 1.429, 0.5, 1.923, 0.422, 0.885, 0.676, 1.136, 1.176, 1.667, 1.299, 1.429, 1.22, 0.662, 0.735, 1.695, 1.562, 0.437, 0.424, 0.893, 1.667, 1.961, 0.571, 0.405, 0.439, 1.205, 0.826, 0.806, 1.064, 0.637, 0.794, 1.471, 1.493, 1.449, 0.781, 0.474, 2.439, 1.136, 1.235, 1.587, 4.762, 0.463, 3.571, 0.99, 0.493, 0.943, 0.498, 1.695, 0.495, 2.273, 0.893, 0.495, 0.943, 1.852, 0.855, 1.587, 2.564, 1.587, 1.064, 0.433, 0.885, 1.852, 2.857, 0.685, 3.846, 0.433, 0.833, 1.333, 0.549, 1.724, 0.478, 1.562, 0.68, 1.852, 1.099, 0.513, 0.806, 0.741, 0.709, 0.556, 0.578, 2.778, 0.962, 0.84, 0.42, 0.452, 1.818, 3.226, 0.543, 9.091, 1.042, 1.19, 0.704, 0.552, 0.971, 1.149, 1.923, 0.917, 1.205, 0.82, 0.741, 2.439, 1.923, 0.441, 0.909, 0, 1.961, 0.971, 0.518, 2.174, 0.685, 0.98, 1.429, 0.613, 2.128, 1.01, 1.075, 0.746, 8.333, 0.431, 7.143, 0.758, 2.222, 0.725, 0.483, 1.333, 1.351, 0.4, 0.588, 0.667, 0.483, 2.083, 0.488, 1.515, 0.926, 2.174, 1.124, 0.885, 8.333, 0.392, 3.448, 2.564, 0.568, 0.758, 1.22, 0.775, 2.128, 1.639, 0.613, 0.463, 0.459, 0.524, 0.625, 0, 0.709, 1.538, 0.433, 0.439, 1.031, 0.971, 0.461, 0.459, 2.041, 5.0, 0.45, 0.704, 1.111, 1.449, 0.4, 0.552, 0.606, 1.667, 0.562, 0.398, 0.508, 0.69, 0.435, 0.758, 0.82, 1.0, 1.299, 1.111, 0.463, 2.174, 1.042, 0.568, 1.282, 0.444, 1.031, 0.448, 0.69, 0.467, 0.452, 1.235, 5.263, 0.413, 2.0, 0.599, 3.704, 1.316, 0.472, 0.424, 1.852, 1.887, 1.449, 0.549, 0.403, 0.422, 0.592, 0.476, 1.667, 0.909, 2.174, 0.599, 0.787, 0.833, 0.671, 0.704, 1.695, 1.887, 1.333, 1.754, 1.266, 0.629, 0, 0.418, 0.943, 1.852, 1.075, 1.471, 0.394, 0.498, 1.818, 0.398, 0.82, 4.0, 7.692, 0.633, 2.0, 0.588, 1.299, 0.49, 1.724, 1.587, 0.585, 0.806, 0.429, 0.488, 2.439, 2.083, 0.704, 0.685, 3.704, 0.585, 1.961, 0.69, 0.543, 0.704, 1.724, 0.676, 1.02, 0.51, 1.493, 0.649, 5.0, 0.637, 0.559, 0.685, 1.136, 1.471, 1.149, 1.515, 2.381, 1.587, 0.752, 4.762, 1.538, 2.857, 1.149, 0.42, 2.0, 0.935, 0.685, 0.885, 0.592, 0.483, 0.952, 0.621, 0.571, 2.632, 1.786, 0.752, 0.467, 0.532, 0.505, 0.763, 0.415, 0.725, 1.136, 1.19, 2.174, 1.064, 2.0, 0.417, 0.556, 0.775, 0.971, 5.0, 0.617, 0.741, 0.667, 4.167, 0.575, 2.128, 2.128, 0.629, 0.398, 0.847, 0.446, 1.266, 0.413, 0.725, 0.694, 0.571, 1.351, 0.781, 1.176, 1.471, 1.053, 1.333, 1.299, 0.463, 1.754, 33.332, 4.167, 0.529, 0.429, 0.413, 4.762, 0.481, 1.724, 0.476, 1.754, 0.971, 0.556, 2.083, 11.111, 1.111, 1.266, 1.449, 1.923, 0.476, 3.333, 0.476, 1.136, 0.521, 0.394, 0.575, 0.617, 1.449, 0.621, 1.316, 0.709, 33.332]


def get_weight():
    return cat_weight,att_weight

def get_feature_vector(coords,feature, WINDOW_SIZE):
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

def set_coords(landmark, visuality):
    v = 0
    coords = []

    if visuality[0] > visuality[1]:
        coords = coords + landmark[0:2]
        centroid[0] += landmark[0]
        centroid[1] += landmark[1]
        v+=1

    if visuality[2] > visuality[3]:
        coords = coords + landmark[2:4]
        centroid[0] += landmark[2]
        centroid[1] += landmark[3]
        v+=1

    if visuality[4] > visuality[5]:
        coords = coords + landmark[4:6]
        centroid[0] += landmark[4]
        centroid[1] += landmark[5]
        v+=1

    if visuality[6] > visuality[7]:
        coords = coords + landmark[6:8]
        centroid[0] += landmark[6]
        centroid[1] += landmark[7]
        v+=1

    if visuality[12] > visuality[13]:
        coords = coords + landmark[12:14]
        centroid[0] += landmark[12]
        centroid[1] += landmark[13]
        v+=1

    if visuality[14] > visuality[15]:
        coords = coords + landmark[14:16]
        centroid[0] += landmark[14]
        centroid[1] += landmark[15]
        v+=1

    if v != 0:
        centroid[0] = centroid[0] / v
        centroid[1] = centroid[1] / v

    if v > 8:
        landmark[8:10] = [-1,-1] #centroid
        landmark[10:12] = [-1,-1]

    return coords


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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_'+filename)

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

def adjust_learning_rate(optimizer, lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.333 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def distance(output,target):
    """Computes the precision@k for distances of the landmarks"""
    dist_function = nn.PairwiseDistance(p=1)
    distances = dist_function(output, target)
    return distances.mean() / 2

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

def f1_score(output,target):
    output = output.data.ge(0).float()
    tp = torch.sum(torch.add(target,output).eq(2))
    fp = torch.sum(torch.add(target,torch.neg(output)).eq(-1))
    fn = torch.sum(torch.add(torch.neg(target),output).eq(-1))
    re = tp/(tp+fn)
    pr = tp/(tp+fp)
    return 2*(re*pr/(re+pr+0.000001)), pr, re 

def get_n_classes(path):
    """get the number of classes"""
    Nclass = 0
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path,item)):
            Nclass = Nclass + 1
    return Nclass

