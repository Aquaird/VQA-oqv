import json
import h5py
import numpy as np
import sys

feature=h5py.File('../../../C3D/sub_activitynet_v1-3.c3d.hdf5','r')

video_list=json.load(open('../video_l.json','r'))
video_h5=[]
count=0
for video in video_list:
    video_h5.append(feature[video]['c3d_features'][:])
    count+=1
    sys.stdout.write('\r%d videos are added.'%count)
    sys.stdout.flush()

np_video_h5=np.array(video_h5)
f=h5py.File('video_feature.h5','w')
f['data']=np_video_h5
f.close()
