import os
import shutil
import torch
import cv2
import numpy as np
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
classes = ['Discussion','Eating','Greeting', 'Phoning',
			 'Purchases','Sitting', 'Posing',
			 'Smoking','Waiting', 'Walking','WalkTogether']#'Direction',

folders = os.listdir('.')
# os.chdir('human3.6')


# for item in classes:
	# os.mkdir('processed/%s'%item)

'''
train_folders = []
for item in folders:
    if 'S' in item:
        train_folders.append(item)

for cs in classes:
	count = 0
	for item in train_folders:
		files = os.listdir('%s/Videos'%item)
		for file in files:
			if cs in file:
				# os.system('cp %s/Videos/%s processed/%s/'%(item,file,cs))
				shutil.copy('%s/Videos/%s'%(item,file),'processed/%s/%s.mp4'%(cs,cs+'_%d'%count))
				count +=1

	print(cs + " done!")


for cs in classes:
	files = os.listdir('processed/%s/'%cs)
	print('%s %d instances'%(cs,len(files)))
'''

### --------------------Load Videos-----------------------------------------------

def load_video(index,clss):
    path = os.path.join('processed/%s/%s_%d.mp4'%(clss,clss,index))
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    vframes = []
    while success:
        vframes.append(torch.Tensor(resize(image,(64,64))))#image/255)#
        success,image = vidcap.read()
    if len(vframes) == 0:
    	return vframes
    return torch.stack(vframes)


for cs in classes:
	arr = []
	for i in range(100):
		length = len(os.listdir('processed/%s'%cs))
		ind = np.random.randint(length)
		frames = load_video(ind,cs)
		if len(frames) == 0:
			i -=1
			continue
		else:
			torch.save(frames,'processed/%s/ptdumps/%d.pt'%(cs,i))
			arr.append(len(frames))
			print('%s %d done'%(cs,i))
	print('%s %d length of frames %d min length'%(cs,np.mean(arr),np.min(arr)))


