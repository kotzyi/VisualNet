import pandas as pd
import re
import random

image_paths = pd.read_table('list_landmarks_inshop.txt',sep="\s+")['image_name'].values.tolist()
classes = {}

def make_triplet_list(num,filename):	
	with open(filename,"w") as fp:
		for i in range(num):
			anchor_key = random.choice(list(classes))
			while len(classes[anchor_key]) < 2:
				anchor_key = random.choice(list(classes))
			positive_key = anchor_key
			negative_key = random.choice(list(classes))
			while anchor_key == negative_key:
				negative_key = random.choice(list(classes))
		
			anchor_id = random.choice(classes[anchor_key])
			positive_id = random.choice(classes[positive_key])
			negative_id = random.choice(classes[negative_key])
			while anchor_id == positive_id:
				positive_id = random.choice(classes[positive_key])
		
			fp.write(str(anchor_id)+" "+str(positive_id)+" "+str(negative_id)+"\n")


for idx, path in enumerate(image_paths):
	parsed_path = re.split("[/]|[_]",path)
	key = "".join(parsed_path[:-2])
	if key in classes:
		classes[key].append(idx)
	else:
		classes[key] =[idx]


make_triplet_list(200000,"train.txt")
