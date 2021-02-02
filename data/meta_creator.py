import os
import torch
import numpy as np

start_dir = 'processed/'
classes = os.listdir(start_dir)

for clas in classes:
	files = os.listdir(start_dir + clas)
	files.sort()
	lists = []
	for i in range(2):
	# for file in files:
		file = files[i]
		if 'train' not in file:
			dic = {}
			dic['vid'] = file
			temp = os.listdir(start_dir + clas+'/'+file)
			temp.sort()
			dic['files'] = np.array_split(temp,4)
			dic['n'] = len(temp)
			lists.append(dic)
	torch.save(lists,'%s/%s/train_meta64x64.pt'%(start_dir,clas))
