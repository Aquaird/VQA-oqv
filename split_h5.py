import h5py
import json
import numpy as np

anno = json.load(open('./dataset/activityNet/qa_anno/activityNet_qa_40.json', 'r'))

c3d_dset = h5py.File('./dataset/activityNet/features_h5/c3d_feature_20.h5', 'r')['c3d']
vgg_dset = h5py.File('./dataset/activityNet/features_h5/vgg_feature_20.h5', 'r')['vgg']
obj_dset = h5py.File('/data/mm/dataset/activityNet/ob_20.h5', 'r')['obj']

all_file = h5py.File('./dataset/activityNet/features_h5/all_20.h5', 'w')
dset_train = all_file.create_dataset('training', shape=(0,20,7,4100), maxshape=(None,20,7,4100))
dset_valid = all_file.create_dataset('validation', shape=(0,20,7,4100), maxshape=(None,20,7,4100))
dset_test = all_file.create_dataset('testing', shape=(0,20,7,4100), maxshape=(None,20,7,4100))

train_index = 0
valid_index = 0
test_index = 0

train_anno = {}
test_anno = {}
valid_anno  = {}

count = 0
for i in anno:
    if anno[i]['split'] == 'training':
        train_anno[i] = anno[i]
        tmp = np.concatenate(
            (
                np.reshape(np.pad(vgg_dset[count], ((0,0),(0,4)), 'constant'),  (20,1,4100)),
                np.reshape(np.pad(c3d_dset[count], ((0,0),(0,4)), 'constant'),  (20,1,4100)),
                obj_dset[count],
            ),
            axis=-2
            )
        train_index += 1
        dset_train.resize(train_index, axis=0)
        dset_train[-1] = tmp
        print('training: ', i, count, train_index)

    if anno[i]['split'] == 'validation':
        valid_anno[i] = anno[i]
        tmp = np.concatenate(
            (
                np.reshape(np.pad(vgg_dset[count], ((0,0),(0,4)), 'constant'),  (20,1,4100)),
                np.reshape(np.pad(c3d_dset[count], ((0,0),(0,4)), 'constant'),  (20,1,4100)),
                obj_dset[count],
            ),
            axis=-2
            )
        valid_index += 1
        dset_valid.resize(valid_index, axis=0)
        dset_valid[-1] = tmp
        print("validation: ", i, count, valid_index)

    if anno[i]['split'] == 'testing':
        test_anno[i] = anno[i]
        tmp = np.concatenate(
            (
                np.reshape(np.pad(vgg_dset[count], ((0,0),(0,4)), 'constant'),  (20,1,4100)),
                np.reshape(np.pad(c3d_dset[count], ((0,0),(0,4)), 'constant'),  (20,1,4100)),
                obj_dset[count],
            ),
            axis=-2
            )
        test_index += 1
        dset_test.resize(test_index, axis=0)
        dset_test[-1] = tmp
        print('testing: ', i, count, test_index)
    count += 1

print(len(train_anno), len(dset_train))
print(len(valid_anno), len(dset_valid))
print(len(test_anno), len(dset_test))

json.dump(train_anno, open('./dataset/activityNet/qa_anno/train_anno.json', 'w'))
json.dump(valid_anno, open('./dataset/activityNet/qa_anno/valid_anno.json', 'w'))
json.dump(test_anno, open('./dataset/activityNet/qa_anno/test_anno.json', 'w'))

all_file.close()
