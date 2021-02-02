import torch
import numpy as np
import os

# classes = #

class Human(object):
	"""docstring for NTUTwoBody"""
	def __init__(self, data_root, seq_len, image_size = 64, train = True):
		super(Human, self).__init__()
		self.data_root = '%s' % data_root
		self.seq_len = seq_len
		self.image_size = image_size
		self.train = train 
		self.classes = ['Direction','Discussion','Eating','Greeting', 'Phoning',
			 'Purchases','Sitting', 'Posing',
			 'Smoking','Waiting', 'Walking','WalkTogether']
		self.target = {'Direction':0,'Discussion':1,'Eating':2,'Greeting':3, 'Phoning':4,
			 'Purchases':5,'Sitting':6, 'Posing':7,
			 'Smoking':8,'Waiting':9, 'Walking':10,'WalkTogether':11}
        # self.dirs = os.listdir(self.data_root)
        

        

	def get_sequence(self):
		t = self.seq_len
		# while True: # skip seqeunces that are too short
		c_idx = np.random.randint(len(self.classes))
		c = self.classes[c_idx]
		# f_idx = np.random.randint(99)
		# dname = '%s/processed/%s/ptdumps/%d.pt' % (self.data_root, c, f_idx)
		while True:
			f_idx = np.random.randint(99)
			dname = '%s/processed/%s/ptdumps/%d.pt' % (self.data_root, c, f_idx)
			if os.path.isfile(dname):
				break
		    # vid_idx = np.random.randint(len(self.data[c]))
		    # vid = self.data[c][vid_idx]
		    # seq_idx = np.random.randint(300)
		    # if len(vid['files'][seq_idx]) - t >= 0:
		    #     break
		# dname = '%s/processed/%s/ptdumps/%d.pt' % (self.data_root, c, f_idx)
		st = np.random.randint(301,600)

		frames = torch.load(dname)
		seq = frames[st:st+t]
        # seq = [] 
        # for i in range(st, st+t):
        #     fname = '%s/%s' % (dname, vid['files'][seq_idx][i])
        #     im = misc.imread(fname)/255.
        #     seq.append(im[:, :, 0].reshape(self.image_size, self.image_size, 1))
		return seq, self.target[c]

	def get_sequence_test(self):
		t = self.seq_len
		# while True: # skip seqeunces that are too short
		c_idx = np.random.randint(len(self.classes))
		c = self.classes[c_idx]
		# f_idx = np.random.randint(99)
		# dname = '%s/processed/%s/ptdumps/%d.pt' % (self.data_root, c, f_idx)
		while True:
			f_idx = np.random.randint(99)
			dname = '%s/processed/%s/ptdumps/%d.pt' % (self.data_root, c, f_idx)
			if os.path.isfile(dname):
				break
		    # vid_idx = np.random.randint(len(self.data[c]))
		    # vid = self.data[c][vid_idx]
		    # seq_idx = np.random.randint(300)
		    # if len(vid['files'][seq_idx]) - t >= 0:
		    #     break
		# dname = '%s/processed/%s/ptdumps/%d.pt' % (self.data_root, c, f_idx)
		st = np.random.randint(301,600)

		frames = torch.load(dname)
		seq = frames[st:st+t]
        # seq = [] 
        # for i in range(st, st+t):
        #     fname = '%s/%s' % (dname, vid['files'][seq_idx][i])
        #     im = misc.imread(fname)/255.
        #     seq.append(im[:, :, 0].reshape(self.image_size, self.image_size, 1))
		return seq, self.target[c]


	def __getitem__(self, index):
		# if not self.seed_set:
			# self.seed_set = True
		# random.seed(index)
		np.random.seed(index)
    #torch.manual_seed(index)
		if self.train:
			x,y = self.get_sequence()
		else:
			x, y = self.get_sequence_test()
		return x, y

	def __len__(self):
		return 36*5 # arbitrary



