import os
import torch

class Model_Manage():
	"""
	This class manages models that can be loaded from landmark_train.py
	"""
	def __init__(self):
		self.models = {}

	def __load_module(self, pkg_name, module_name):
		module = __import__("{0}.{1}".format(pkg_name, module_name), globals(), locals(), [module_name])
		return module

	def chkpoint_load(self, model, model_params, save_file_path):
		try:
			params = {}
			if os.path.isfile(save_file_path):
				print("Loading checkpoint '{}'".format(save_file_path))
				checkpoint = torch.load(save_file_path)
				for p in model_params:
					params[p] = checkpoint[p]

				model.load_state_dict(checkpoint['state_dict'])
				print("Loaded checkpoint in {}".format(save_file_path))
				return model, params
			else:
				print("Can't load save file")
				return model, None
		except:
			print("Can't load model")
			return None, None

	def add(self, args, CUDA=True, isPretrained=True):
		network_file, arch = args
		network = self.__load_module('models', network_file.replace('.py',''))
		model = network.make_model(arch, pretrained=isPretrained)
		if CUDA:
			model = torch.nn.DataParallel(model).cuda()

		print("Added '{0}' model in '{1}'".format(arch, network_file))
		return model 
