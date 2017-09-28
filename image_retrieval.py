import pickle

def pickle_load(filename):
	path = []
	data = []
	with open(filename,'rb') as fp:
		while True:
			try:
				p, d = pickle.load(fp)
				path.append(p)
				data.append(d)
			except EOFError:
				break

	return path,data


path, data = pickle_load('output.pickle')
print(path)
