import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import socket
from tqdm import tqdm
import functionals_pkg.batch_fns as bf
from data_pkg import data_fns as df
from functionals_pkg import argparse_fns as af
from sys import argv
import model_pkg.models as models
metric = af.getopts(argv)

n_gpus = 1
server=0

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


print "############################"
print "SET UP VIDEO STREAM"
print "############################"

#Get Data Stream
train_test = 'Train'

size_axis = 32
n_frames = 5

strides = int(metric['-strd'])
gs = bool(int(metric['-gs']))
tstrides = int(metric['-tstrd'])
lstm = bool(int(metric['-lstm']))

if(lstm):
    ts = 'first'
    conv = 'lstm'
    model = models.Conv_LSTM_autoencoder_nostream(model_store = 'stats/lstm_tstrd_'+str(tstrides)+'_stats',loss='bce',h_units=16,n_timesteps=5,n_channels=1,
                                        batch_size=72,n_clusters=10, clustering_lr=1, lr_model=1e-4,lamda=0.01,
                                        lamda_assign=0.01,n_gpus=n_gpus,gs=gs,notrain=False,data_folder='folder',reverse=False)
    ax = 0
else:
    ts = 'last'
    conv = 'conv'
    ax = -1
    model = models.Conv_autoencoder_nostream(model_store='stats/conv_tstrd_'+str(tstrides)+'_stats',loss='bce',h_units=16,n_timesteps=5,n_channels=1,
                                        batch_size=72,n_clusters=10, clustering_lr=1, lr_model=1e-4,lamda=0.01,
                                        lamda_assign=0.01,n_gpus=n_gpus,gs=gs,notrain=False,data_folder='folder',reverse=False)


vstream = df.Video_Stream_ARTIF(video_path=path_videos, video_train_test=train_test, size_y=size_axis, size_x=size_axis,
                                timesteps=n_frames,ts_first_or_last=ts,strides=strides,tstrides=tstrides)


print "#############################"
print "SET-UP LIST OF MOTION HISTOGRAMS"
print "#############################"

list_mot_hist=[]


print "START_OF_DATA"
for k in tqdm(range(0, len(vstream.seek_dict))):
    cubatch = bf.get_next_relevant_cuboids(vstream, gs=gs)

    if(gs==0):
        cubatch = bf.norm_batch(cubatch)

    variances = bf.get_variances(cubatch,ax,gs)
    list_mot_hist.append(variances.tolist())

    del cubatch
    del variances

print "END_OF_DATA"

#Flatten the list
flat_list_mot_hist = [item for sublist in list_mot_hist for item in sublist]

max_variance = max(flat_list_mot_hist)
min_variance = min(flat_list_mot_hist)
mean_variance = np.mean(flat_list_mot_hist)
std_variance = np.std(flat_list_mot_hist)

print "########################"
print "MAX VARIANCE:"
print max_variance
print "########################"

print "########################"
print "MIN VARIANCE:"
print min_variance
print "########################"

print "########################"
print "MEAN VARIANCE:"
print mean_variance
print "########################"

print "########################"
print "STD VARIANCE:"
print std_variance
print "########################"

print "########################"
print "HISTOGRAM"
print "########################"

hist,bins = np.histogram(flat_list_mot_hist,bins=100)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
print "########################"
print "CENTERS : ",center
print "########################"
print "########################"
print "HIST : ",hist
print "########################"

plt.bar(center, hist, align='center', width=width)
plt.ylabel('Variance mean')
plt.savefig('artif_variance_mean_prob_'+str(tstrides)+'_'+conv+'.png', bbox_inches='tight')
plt.close()



center = center.tolist()
center_gif_done = []

for _ in center:
    center_gif_done.append(0)


center.append(np.inf)
vstream.reset_stream()

print "START_OF_DATA"
while any(v==0 for v in center_gif_done):

    cubatch = bf.get_next_relevant_cuboids(vstream, gs=gs)

    if(gs==0):
        cubatch = bf.norm_batch(cubatch)

    variances = bf.get_variances(cubatch,ax,gs)

    for idx,j in enumerate(variances):
        for k in range(0,len(center)-1):
            if(j >= center[k]) and (j<=center[k+1]) and center_gif_done[k]==0:
                center_gif_done[k]=1
                model.save_gifs(cubatch[idx],str(center[k])+'_'+str(center[k+1]))

    del cubatch
    del variances

print "END_OF_DATA"