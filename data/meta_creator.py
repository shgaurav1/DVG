import os
import numpy as np
import json

start_dir = 'data/processed/'
classes = os.listdir(start_dir)

for clas in classes:
    files = os.listdir(start_dir + clas)
    files.sort()
    train = [f"person{i:02}_" for i in range(0, 17)]
    test = [f"person{i:02}_" for i in range(17, 26)]
    data = {"train": [], "test": []}
    for file in files:
        if 'train' in file or 'test' in file:
            continue

        temp = os.listdir(start_dir + clas+'/'+file)
        temp.sort()
        dic = {
            'vid': file,
            'files': [i.tolist() for i in np.array_split(temp, 4)],
            'n': len(temp)
        }
        if any([person in file for person in train]):
            data['train'].append(dic)
        elif any([person in file for person in test]):
            data['test'].append(dic)
        else:
            raise ValueError(f"Expected {file} to be in train or test set.")

    for prefix, lists in data.items():
        file = f'{start_dir}/{clas}/{prefix}_meta64x64.json'
        print(f"Saving to {file}")
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(lists, f, ensure_ascii=False, indent=4)
