import os
import json
import caffe
import multiprocessing as mp
import threading
import time
import sys
import shutil

def gen_clips_frames(param):
    anno_json = param['anno_json']
    img_dir = param['img_dir']
    clips_dir = param['clip_dir']
    start_end = anno_json['start_end']
    start = float(start_end[1:-1].strip().split(',')[0])
    end = float(start_end[1:-1].strip().split(',')[1])

    start_frame = int(start * 25)
    end_frame = int(end * 25) + 1

    for i in range(start_frame, end_frame):
        file_name = 'image_'+str(i).zfill(5)+'.jpg'
        shutil.copy(os.path.join(img_dir,file_name),os.path.join(clips_dir,file_name))

if __name__ == "__main__":

    t0 = time.time()
    annos = json.load(open("../dataset/activityNet/qa_anno/activity_net_qa.json", 'r'))
    img_root = '/home/mm/dataset/activityNet/frames'
    clip_root = '../dataset/activityNet/clips_frames'
    training_list = os.listdir(os.path.join(img_root, 'training'))
    validation_list = os.listdir(os.path.join(img_root, 'validation'))

    print(len(training_list), len(validation_list))
    param_list = []

    pool = mp.Pool(12)
    for idx, anno in annos.items():
        if len(anno['video']) == 11:
            video_id = anno['video']
        else:
            video_id = anno['video'][2:]

        if video_id in training_list:
            video_img_dir = os.path.join(os.path.join(img_root, 'training'), video_id)
        elif video_id in validation_list:
            video_img_dir = os.path.join(os.path.join(img_root, 'validation'), video_id)
        else:
            print('video: ' + video_id + ' not downloaded')
            continue

        clip_img_dir = os.path.join(clip_root, idx)
        if not os.path.exists(clip_img_dir):
            os.mkdir(clip_img_dir)

        param_list.append({'anno_json':anno, 'img_dir':video_img_dir, 'clip_dir':clip_img_dir})

    print('parm_list completed!', time.time()-t0)
    print(str(len(param_list)) + '/' + str(len(annos)))

    pool.map(gen_clips_frames, param_list)
    pool.close()
    pool.join()
