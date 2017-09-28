import pickle
from sklearn.cluster import KMeans
import numpy as np

class Clustering():
	def __init__(self, filename, nCluster):
		self.filename = filename
		self.nCluster = nCluster
		self.raw = self.pickle_load(filename)
		self.paths,self.coords = self.make_list(self.raw)
		self.clusters = self.make_cluster(self.paths, self.coords, nCluster)


	def pickle_load(self, filename):
		data = []
		with open(filename,'rb') as fp:
			while True:
				try:
					p, d = pickle.load(fp)
					data.append((p,d))
				except EOFError:
					break
		
		return data

	def make_list(self, raw):
		coords_data = []
		paths_data = []
		for paths, coords in raw:	
			for (path,coord) in zip(paths,coords):
				paths_data.append(path)
				coords_data.append(coord)

		return paths_data, coords_data

	def make_cluster(self, paths, coords, nCluster):
		clusters = {}
		coords = np.array(coords)

		kmeans = KMeans(n_clusters = nCluster, random_state = 0, max_iter = 1000).fit(coords)
		for (label, path) in zip(kmeans.labels_, paths):
			if label in list(clusters.keys()):
				clusters[label].append(path)
			else:
				clusters[label] = []

		return clusters

def gen_cluster(filename,nCluster):
	cluster = Clustering(filename,nCluster)
	return cluster
