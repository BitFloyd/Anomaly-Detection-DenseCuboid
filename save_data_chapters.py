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


metric = af.getopts(argv)

if(socket.gethostname()=='puck'):
    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/art_videos_prob_0.01/artif_videos_128x128'

elif('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/scratch/suu-621-aa/ANOMALY/densecub')
    path_videos='/scratch/suu-621-aa/ANOMALY/data/art_videos_prob_0.01/artif_videos_128x128'

else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/densecub/')
    path_videos = '/gs/project/suu-621-aa/sejacob/data/art_videos_prob_0.01/artif_videos_128x128'


strides = int(metric['-strd'])
gs = bool(int(metric['-gs']))
tstrides = int(metric['-tstrd'])
lstm = bool(int(metric['-lstm']))
filename = 'chapters_tstrd_'+str(tstrides)+'_gs_'+str(gs)+'_lstm_'+str(lstm)+'.txt'
if(lstm):
    ts = 'first'
    ts_pos = 0
    mainfol = 'chapter_store_lstm'

else:
    ts = 'last'
    ts_pos = -1
    mainfol = 'chapter_store_conv'

tv = 0.0

if(gs):
    folder = os.path.join(mainfol, 'data_store_greyscale_' + str(tstrides))
    if(tstrides==2):
        # tv = 0.02
        print "VARIANCE THRESHOLD IS:", tv
    elif(tstrides==4):
        # tv = 0.05
        print "VARIANCE THRESHOLD IS:", tv
    elif (tstrides==8):
        # tv = 0.05
        print "VARIANCE THRESHOLD IS:", tv

else:
    # tv = 0.0
    folder = os.path.join(mainfol, 'data_store_' + str(tstrides) + '_' + str(tv))


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
                                timesteps=n_frames,ts_first_or_last=ts,strides=strides,tstrides=tstrides)


print "############################"
print "ASSEMBLE AND SAVE DATASET"
print "############################"

if(os.path.exists(os.path.join(folder,'vid_this.npy'))):
    video_id = np.load(os.path.join(folder,'vid_this.npy'))
    chapter_id = np.load(os.path.join(folder,'chap_this.npy'))
else:
    video_id = np.array([0])
    chapter_id = np.array([0])


print "SAVING FOR VIDEO ID:"
print video_id
print "STARTING WITH CHAPTER ID: ",chapter_id

vstream.set_seek_to_video(video_id[0])

print "SEEK IS NOW:", vstream.seek


while True:

    list_cubatch = []

    for k in tqdm(range(0,200)):
        cubatch = bf.return_relevant_cubs(vstream,thresh_variance=tv,gs=gs,ts_pos=ts_pos)
        list_cubatch.append(cubatch.tolist())

        if(vstream.seek+1 == len(vstream.seek_dict.values())):
            break

        if (len(vstream.seek_dict[vstream.seek+1]) > 1):
            break

    flat_list_cubs = [item for sublist in list_cubatch for item in sublist]


    print "LENGTH OF VIDEO ARRAY"
    print len(flat_list_cubs)

    length = len(flat_list_cubs)

    if(len(flat_list_cubs))==0:
        print "EMPTY ARRAY"
        sys.exit()

    print "DELETING LIST_CUBATCH"
    del (list_cubatch[:])

    x = np.array(flat_list_cubs)
    print "DELETING FLAT_LIST"
    del(flat_list_cubs[:])


    print "SAVING CHAPTER ID:",str(chapter_id[0])
    f = open(filename,'a+')
    f.write('chapter_id: '+str(chapter_id[0])+'=> '+str(length)+'\n')
    f.close()

    np.save(os.path.join(folder,'chapter_'+str(chapter_id[0])+'.npy'),x)

    print "DELETING ARRAY:"
    del x

    chapter_id+=1
    if(vstream.seek+1 == len(vstream.seek_dict.values())):
        break

    if (len(vstream.seek_dict[vstream.seek+1]) > 1):
        break



video_id+=1
np.save(os.path.join(folder,'vid_this.npy'),np.array(video_id))
np.save(os.path.join(folder,'chap_this.npy'),np.array(chapter_id))