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

path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'

strides = int(metric['-strd'])
gs = bool(int(metric['-gs']))
tstrides = int(metric['-tstrd'])
lstm = False

test = 1

if(lstm):
    ts = 'first'
    ts_pos = 0
    mainfol = 'chapter_store_lstm_test'

else:
    ts = 'last'
    ts_pos = -1
    mainfol = 'chapter_store_conv_test'



tv = 0.0

if(gs):
    folder = os.path.join(mainfol,'ucsd2_data_store_greyscale_test_bkgsub'+str(tstrides))

else:
    folder = os.path.join(mainfol,'ucsd2_data_store_test_bkgsub'+str(tstrides))

if(not os.path.exists(mainfol)):
    os.mkdir(mainfol)

if(not os.path.exists(folder)):
    os.mkdir(folder)


print "############################"
print "SET UP VIDEO STREAM"
print "############################"

#Get Data Stream
train_test = 'Test'

size_axis = 48
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
    list_cuboids_pixmap_full_video = []
    list_cuboids_anomaly_full_video = []
    list_cuboids_anompercentage_full_video = []


    while True:

        anom_stats = False

        list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly,list_cuboids_anompercentage = bf.return_cuboid_test(vstream,thresh_variance=tv,gs=gs,ts_pos=ts_pos)

        if(len(list_cuboids_full_video)):

            list_cuboids_full_video.append(list_cuboids[-1])
            list_cuboids_pixmap_full_video.append(list_cuboids_pixmap[-1])
            list_cuboids_anomaly_full_video.append(list_cuboids_anomaly[-1])
            list_cuboids_anompercentage_full_video.append(list_cuboids_anompercentage[-1])
        else:
            list_cuboids_full_video=list_cuboids
            list_cuboids_pixmap_full_video=list_cuboids_pixmap
            list_cuboids_anomaly_full_video=list_cuboids_anomaly
            list_cuboids_anompercentage_full_video = list_cuboids_anompercentage


        if(vstream.seek+1 == len(vstream.seek_dict.values())):
            break

        if (len(vstream.seek_dict[vstream.seek+1]) > 1):
            break


    video_id += 1

    with h5py.File(os.path.join(folder, 'data_test_video_cuboids.h5'), "a") as f:
        dset = f.create_dataset('video_cuboids_array_'+str(video_id), data=np.array(list_cuboids_full_video))
        print(dset.shape)

    with h5py.File(os.path.join(folder, 'data_test_video_pixmap.h5'), "a") as f:
        dset = f.create_dataset('video_pixmap_array_'+str(video_id), data=np.array(list_cuboids_pixmap_full_video))
        print(dset.shape)

    with h5py.File(os.path.join(folder, 'data_test_video_anomgt.h5'), "a") as f:
        dset = f.create_dataset('video_anomgt_array_'+str(video_id), data=np.array(list_cuboids_anomaly_full_video))
        print(dset.shape)

    with h5py.File(os.path.join(folder, 'data_test_video_anomperc.h5'), "a") as f:
        dset = f.create_dataset('video_anomperc_array_'+str(video_id), data=np.array(list_cuboids_anompercentage_full_video))
        print(dset.shape)

    del list_cuboids_full_video
    del list_cuboids_pixmap_full_video
    del list_cuboids_anomaly_full_video
    del list_cuboids_anompercentage_full_video


    if(vstream.seek+1 == len(vstream.seek_dict.values())):
        break

print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "FINISHED ALL VIDEOS"
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"