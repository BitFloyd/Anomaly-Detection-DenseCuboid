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

if (socket.gethostname() == 'puck'):
    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2_128x85/Test'

elif ('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/scratch/suu-621-aa/ANOMALY/densecub')
    path_videos = '/scratch/suu-621-aa/ANOMALY/data/art_videos_prob_0.01/artif_videos_128x128'

else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/densecub')
    path_videos = '/gs/project/suu-621-aa/sejacob/data/art_videos_prob_0.01/artif_videos_128x128'


strides = int(metric['-strd'])
gs = bool(int(metric['-gs']))
tstrides = int(metric['-tstrd'])
lstm = bool(int(metric['-lstm']))

test = 1


filename = 'chapters_tstrd_'+str(tstrides)+'_gs_'+str(gs)+'_lstm_'+str(lstm)+'_test_'+str(test)+'.txt'

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
    folder = os.path.join(mainfol,'ucsd_data_store_greyscale_test_bkgsub'+str(tstrides))


else:
    folder = os.path.join(mainfol,'ucsd_data_store_test_bkgsub'+str(tstrides))

if(not os.path.exists(mainfol)):
    os.mkdir(mainfol)

if(not os.path.exists(folder)):
    os.mkdir(folder)

print "############################"
print "SET UP VIDEO STREAM"
print "############################"

#Get Data Stream
train_test = 'Test'

size_axis = 24
n_frames = 8
vstream = df.Video_Stream_ARTIF(video_path=path_videos, video_train_test=train_test, size_y=size_axis, size_x=size_axis,
                                timesteps=n_frames,ts_first_or_last=ts,strides=strides,tstrides=tstrides,anompth=0.1,
                                bkgsub=True)


print "############################"
print "ASSEMBLE AND SAVE DATASET"
print "############################"


if(os.path.exists(os.path.join(folder,'vid_this.npy'))):
    video_id = np.load(os.path.join(folder,'vid_this.npy'))


else:
    video_id = np.array([0])



print "SAVING FOR VIDEO ID:"
print video_id


vstream.set_seek_to_video(video_id[0])

print "SEEK IS NOW:", vstream.seek



list_cuboids_full_video = []
list_cuboids_pixmap_full_video = []
list_cuboids_anomaly_full_video = []

while True:

    anom_stats = False

    list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly = bf.return_cuboid_test(vstream,thresh_variance=tv,gs=gs,ts_pos=ts_pos)

    if(len(list_cuboids_full_video)):

        list_cuboids_full_video.append(list_cuboids[-1])
        list_cuboids_pixmap_full_video.append(list_cuboids_pixmap[-1])
        list_cuboids_anomaly_full_video.append(list_cuboids_anomaly[-1])
    else:
        list_cuboids_full_video=list_cuboids
        list_cuboids_pixmap_full_video=list_cuboids_pixmap
        list_cuboids_anomaly_full_video=list_cuboids_anomaly


    if(vstream.seek+1 == len(vstream.seek_dict.values())):
        break

    if (len(vstream.seek_dict[vstream.seek+1]) > 1):
        break



video_id+=1
np.save(os.path.join(folder,'vid_this.npy'),video_id)

with h5py.File(os.path.join(folder, 'data_test_video_cuboids.h5'), "a") as f:
    dset = f.create_dataset('video_cuboids_array_'+str(video_id[0]), data=np.array(list_cuboids_full_video))
    print(dset.shape)

with h5py.File(os.path.join(folder, 'data_test_video_pixmap.h5'), "a") as f:
    dset = f.create_dataset('video_pixmap_array_'+str(video_id[0]), data=np.array(list_cuboids_pixmap_full_video))
    print(dset.shape)

with h5py.File(os.path.join(folder, 'data_test_video_anomgt.h5'), "a") as f:
    dset = f.create_dataset('video_anomgt_array_'+str(video_id[0]), data=np.array(list_cuboids_anomaly_full_video))
    print(dset.shape)


