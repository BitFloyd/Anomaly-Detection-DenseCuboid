import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from functionals_pkg import argparse_fns as af
from functionals_pkg import batch_fns as bf
from sys import argv
from data_pkg import data_fns as df
import pickle
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
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2_128x85/Test'

elif('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/scratch/suu-621-aa/ANOMALY/densecub')
    path_videos='/scratch/suu-621-aa/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2_128x85/Test'

else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/densecub/')
    path_videos = '/gs/project/suu-621-aa/sejacob/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2_128x85/Test'

strides = int(metric['-strd'])
gs = bool(int(metric['-gs']))
tstrides = int(metric['-tstrd'])
lstm = bool(int(metric['-lstm']))

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
    folder = os.path.join(mainfol, 'analysis_gs')

else:
    folder = os.path.join(mainfol,'analysis')


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
                                timesteps=n_frames,ts_first_or_last=ts,strides=strides,tstrides=tstrides,anompth=0.1,bkgsub=True)


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

while True:

    anom_stats = False

    _, _, _ = bf.return_cuboid_test(vstream,thresh_variance=tv,gs=gs,ts_pos=ts_pos)

    if(vstream.seek+1 == len(vstream.seek_dict.values())):
        break

    if (len(vstream.seek_dict[vstream.seek+1]) > 1):
        break

video_id+=1
np.save(os.path.join(folder,'vid_this.npy'),video_id)

if(os.path.exists('list_anom_percentage.pkl')):
    with open('list_anom_percentage.pkl', 'rb') as f:
        list_anom_percentage = pickle.load(f)
else:
    list_anom_percentage =[]

list_anom_percentage.extend(df.list_anom_percentage)

hist,bins = np.histogram(list_anom_percentage,bins=200)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
print "########################"
print "CENTERS : ",center
print "########################"
print "########################"
print "HIST : ",hist
print "########################"

plt.bar(center, hist, align='center', width=width)
plt.ylabel('Number of samples')
plt.xlabel('Anomaly Percentage')
plt.savefig('anom_percentage_histogram_ucsd2.png', bbox_inches='tight')
plt.close()

with open(os.path.join('list_anom_percentage.pkl'), 'wb') as f:
    pickle.dump(list_anom_percentage, f)