import matplotlib as mpl
mpl.use('Agg')
from functionals_pkg import argparse_fns as af
from functionals_pkg import batch_fns as bf
from sys import argv
from data_pkg import data_fns as df
import os
import socket
import numpy as np
import h5py

metric = af.getopts(argv)

dataset = metric['-dataset']

if(dataset =='ucsd2'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/UCSD2/TRAIN'
elif(dataset=='triangle'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/art_videos_triangle/Test'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/TRIANGLE/TRAIN'
elif(dataset=='ucsd1'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/UCSD1/TRAIN'

elif(dataset=='boat_holborn'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/york/Boat-Holborn/Train'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/BOAT-HOLBORN/TRAIN'

elif(dataset=='boat_sea'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/york/Boat-Sea/Train'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/BOAT-SEA/TRAIN'

elif(dataset=='camouflage'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/york/Camouflage/Train'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/CAMOUFLAGE/TRAIN'

elif(dataset=='canoe'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/york/Canoe/Train'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/CANOE/TRAIN'

elif(dataset=='traffic_train'):
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/york/Traffic-Train/Train'
    save_folder = '/usr/local/data/sejacob/ANOMALY/densecub/DATA/TRAFFIC-TRAIN/TRAIN'


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

while True:
    vstream = df.Video_Stream_ARTIF(video_path=path_videos, video_train_test=train_test, size_y=size_axis, size_x=size_axis,
                                    timesteps=n_frames,ts_first_or_last=ts,strides=strides,tstrides=tstrides,anompth=0.20,
                                    bkgsub=True)


    print "############################"
    print "ASSEMBLE AND SAVE DATASET"
    print "############################"

    print "SAVING FOR VIDEO ID:"
    print video_id


    vstream.set_seek_to_video(video_id)

    print "SEEK IS NOW:", vstream.seek



    list_cuboids_full_video = []



    while True:

        anom_stats = False

        list_cuboids, _, _,_ = bf.return_cuboid_test(vstream,thresh_variance=tv,gs=gs,ts_pos=ts_pos)

        if(len(list_cuboids_full_video)):

            list_cuboids_full_video.append(list_cuboids[-1])

        else:
            list_cuboids_full_video=list_cuboids


        if(vstream.seek+1 == len(vstream.seek_dict.values())):
            break

        if (len(vstream.seek_dict[vstream.seek+1]) > 1):
            break


    video_id+=1

    with h5py.File(os.path.join(save_folder, 'data_train_video_cuboids.h5'), "a") as f:
        dset = f.create_dataset('video_cuboids_array_'+str(video_id), data=np.array(list_cuboids_full_video))
        print(dset.shape)

    del list_cuboids_full_video

    if (vstream.seek + 1 == len(vstream.seek_dict.values())):
        break


print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "FINISHED ALL VIDEOS"
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"