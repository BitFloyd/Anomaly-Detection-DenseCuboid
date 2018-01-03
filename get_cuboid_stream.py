import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from functionals_pkg import feature_fns as ff
from sys import argv
from data_pkg import data_fns as df
from tqdm import tqdm
import functionals_pkg.config as cfg
import os
import socket
import numpy as np


lamda = cfg.lamda
lamda_assign = cfg.lamda_assign

if(socket.gethostname()=='puck'):

    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"
    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    #
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # set_session(tf.Session(config=config))

elif('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/scratch/suu-621-aa/ANOMALY/densecub')
    lamda = cfg.lamda_helios
    lamda_assign=cfg.lamda_helios_assign

else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/densecub/')


metric = af.getopts(argv)
print metric

#
# # Get MODEL
# model_store = 'models/convlstm2d_autoencoder'
#
# ae_model = models.Conv_LSTM_autoencoder(model_store=model_store,loss='bce',h_units=int(metric['-h']),n_timesteps=5,batch_size=128,n_clusters=10, clustering_lr=1, lr_model=1e-4,lamda=lamda,lamda_assign=lamda_assign)
#


print "############################"
print "SET UP VIDEO STREAM"
print "############################"

#Get Data Stream
train_test = 'Train'

path_videos = '/usr/local/data/sejacob/ANOMALY/data/art_videos_prob_0.01/artif_videos_128x128'

size_axis = 32
n_frames = 5
vstream = df.Video_Stream_ARTIF(video_path=path_videos, video_train_test=train_test, size_y=size_axis, size_x=size_axis,timesteps=n_frames,ts_first_or_last='first',strides=2)


# ae_model.change_clustering_weight(0.0)
# ae_model.ae.summary()

# list_random_for_km = []
print "#############################"
print "SET-UP LIST OF MOTION HISTOGRAMS"
print "#############################"

list_mot_hist=[]


print "START_OF_EPOCH"
for k in tqdm(range(0, len(vstream.seek_dict))):
    _, _, _, cuboids_batch, _ = vstream.get_next_cuboids_from_stream()
    list_mot_hist.append(ff.get_variances(cuboids_batch,vstream))
    del(cuboids_batch)


print "END OF EPOCH"

#Flatten the list
flat_list_mot_hist = [item for sublist in list_mot_hist for item in sublist]

print "########################"
print "HISTOGRAM"
print "########################"

hist,bins = np.histogram(flat_list_mot_hist,bins=40,normed=False)

print "########################"
print "BINS : ",bins
print "########################"
print "########################"
print "HIST : ",hist
print "########################"

plt.hist(flat_list_mot_hist, normed=False, bins=40)
plt.ylabel('Variance sum prob')
plt.savefig('variance_sum_artif.png', bbox_inches='tight')
plt.close()




