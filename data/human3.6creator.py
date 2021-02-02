import os

classes = ['Direction','Discussion','Eating','Greeting', 'PhoneCall',
			 'Purchases','Sitting', 'SittingDown', 'TakingPhoto','Posing',
			 'Smoking','Waiting', 'Walking','WalkingTogether','WalkingDog']

folders = os.listdir('.')
# os.chdir('human3.6')


for item in classes:
	os.mkdir('processed/%s'%item)


train_folders = []
for item in folders:
    if 'S' in item:
        train_folders.append(item)


for item in train_folders:
	files = os.listdir('%s/Videos'%item)
	for cs in classes:
		for file in files:
			if cs in file:
				os.system('cp %s/Videos/%s processed/%s/%s'%(item,file,cs,file))