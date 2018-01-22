import h5py
import json
from PIL import Image
import os

import numpy as np

f = json.load(open('./dataset/activityNet/qa_anno/activityNet_qa_40.json'))
h5file = h5py.File('./dataset/clips_raw.hdf5', 'w')
dset = h5file.create_dataset('224', shape=(0, 224, 224, 3), maxshape=(None, 224, 224, 3), dtype=np.uint8)
video_code = '0'
tmp = list()
all_number = len(f)
count = 0
for i in f:
    print('procesing: '+i)
    print(100*int(i)/all_number)
    if f[i]['video_code'] != video_code:
        for idx in range(f[i]['start'], f[i]['end']):
            path = '/data/'+f[i]['frame_path'][6:]
            img = Image.open(os.path.join(path, 'image_'+str(int(idx)+1).zfill(5)+'.jpg'))
            img = np.array(img.resize((224,224), Image.BILINEAR))
            tmp.append(img)
        video_code = f[i]['video_code']
        dset.resize(count+len(tmp), axis=0)
        dset[count:] = tmp
        count += len(tmp)
        tmp = list()
    else:
        continue

h5file.close()
print(count, video_code)
all_frames = 0
vodeo_code = '0'
for i in f:
    if f[i]['video_code'] != video_code:
        all_frames += (f[i]['end']-f[i]['start'])
        video_code = f[i]['video_code']
    else:
        continue

print(all_frames)
