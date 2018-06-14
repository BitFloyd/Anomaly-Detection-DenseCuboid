import matplotlib as mpl
mpl.use('Agg')
from functionals_pkg import argparse_fns as af
from functionals_pkg import batch_fns as bf
from sys import argv
from data_pkg import data_fns as df
from tqdm import tqdm
import os
import socket
import numpy as np
import sys
import h5py
import time


metric = af.getopts(argv)
dataset = metric['-dataset']

if(dataset =='ucsd2'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/UCSD2/TRAIN'
elif(dataset=='triangle'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/art_videos_triangle/Train'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/TRIANGLE/TRAIN'
elif(dataset=='ucsd1'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/UCSD1/TRAIN'

strides = int(metric['-strd'])
gs = bool(int(metric['-gs']))
tstrides = int(metric['-tstrd'])
size = int(metric['-size'])

ts = 'last'
ts_pos = -1

tv = 0.0


if(not os.path.exists(save_folder)):
    os.mkdir(save_folder)

print "############################"
print "SET UP VIDEO STREAM"
print "############################"

#Get Data Stream
train_test = 'Train'

size_axis = size
n_frames = 8

video_id = 0
chapter_id = 0

while True:

    vstream = df.Video_Stream_ARTIF(video_path=path_videos, video_train_test=train_test, size_y=size_axis, size_x=size_axis,
                                timesteps=n_frames,ts_first_or_last=ts,strides=strides,tstrides=tstrides,bkgsub=True)


    print "############################"
    print "ASSEMBLE AND SAVE DATASET"
    print "############################"
    print "SAVING FOR VIDEO ID:"
    print video_id
    print "STARTING WITH CHAPTER ID: ",chapter_id

    vstream.set_seek_to_video(video_id)

    print "SEEK IS NOW:", vstream.seek


    list_cubatch = []

    start_time = time.time()
    while True:
        cubatch = bf.return_relevant_cubs(vstream,thresh_variance=tv,gs=gs,ts_pos=ts_pos)
        list_cubatch.append(cubatch.tolist())

        if(vstream.seek+1 == len(vstream.seek_dict.values())):
            break

        if (len(vstream.seek_dict[vstream.seek+1]) > 1):
            break
    end_time = time.time()

    print "TIME_TAKEN: ",(end_time-start_time)/60.0," MINUTES"
    flat_list_cubs = [item for sublist in list_cubatch for item in sublist]


    print "LENGTH OF VIDEO ARRAY"
    print len(flat_list_cubs)

    length = len(flat_list_cubs)

    if(len(flat_list_cubs))==0:
        print "EMPTY ARRAY"
        sys.exit()

    print "DELETING LIST_CUBATCH"
    del (list_cubatch[:])

    with h5py.File(os.path.join(save_folder,'data_train.h5'), "a") as f:
        dset = f.create_dataset('chapter_'+str(chapter_id),data=np.array(flat_list_cubs))
        print(dset.shape)

    del flat_list_cubs
    print "SAVING CHAPTER ID:",str(chapter_id)

    chapter_id+=1
    video_id+=1

    if(vstream.seek+1 == len(vstream.seek_dict.values())):
        break

print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "FINISHED ALL VIDEOS"
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
