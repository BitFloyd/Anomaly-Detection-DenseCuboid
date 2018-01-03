import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from functionals_pkg import batch_fns as bf
from sys import argv
from data_pkg import data_fns as df
from tqdm import tqdm
import functionals_pkg.config as cfg
import os
import socket
import numpy as np


lamda = cfg.lamda
lamda_assign = cfg.lamda_assign

metric = af.getopts(argv)

n_gpus = 1
server=0

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
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/art_videos_prob_0.01/artif_videos_128x128'


elif('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/scratch/suu-621-aa/ANOMALY/densecub')
    path_videos='/scratch/suu-621-aa/ANOMALY/data/art_videos_prob_0.01/artif_videos_128x128'
    n_gpus = int(metric['-ngpu'])


else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/densecub/')
    path_videos = '/gs/project/suu-621-aa/sejacob/data/art_videos_prob_0.01/artif_videos_128x128'
    n_gpus = int(metric['-ngpu'])



h_units = int(metric['-h'])
n_epochs = int(metric['-n'])
strides = int(metric['-strd'])
mini_batch_size=int(metric['-mbs'])

print "############################"
print "SET UP MODEL"
print "############################"

# Get MODEL
model_store = 'models/convlstm2d_autoencoder'

ae_model = models.Conv_LSTM_autoencoder(model_store=model_store,loss='mse',h_units=h_units,n_timesteps=5,
                                        batch_size=128,n_clusters=10, clustering_lr=1, lr_model=1e-4,lamda=lamda,
                                        lamda_assign=lamda_assign,n_gpus=n_gpus)



print "############################"
print "SET UP VIDEO STREAM"
print "############################"

#Get Data Stream
train_test = 'Train'

size_axis = 32
n_frames = 5
vstream = df.Video_Stream_ARTIF(video_path=path_videos, video_train_test=train_test, size_y=size_axis, size_x=size_axis,
                                timesteps=n_frames,ts_first_or_last='first',strides=strides)

if(int(metric['-test'])):
    length_run = int(metric['-test'])
else:
    length_run = len(vstream.seek_dict)


ae_model.change_clustering_weight(0.0)

print "############################"
print "START_OF_EPOCH - 1"
print "############################"

for k in tqdm(range(0, length_run)):
    bf.process_a_batch(ae_model,vstream,thresh_variance=0.1,mini_batch_size=mini_batch_size,train=False)

ae_model.save_weights()
vstream.reset_stream()
bf.do_recon(ae_model,vstream,thresh_variance=0.1,num_recons=10)
ae_model.generate_loss_graph('epoch1_loss_graph.png')
print "############################"
print "END OF EPOCH - 1"
print "############################"

print "############################"
print "START_OF_EPOCH K MEANS"
print "############################"

for k in tqdm(range(0, length_run)):
    bf.process_k_means_a_batch(ae_model,vstream,thresh_variance=0.1,mini_batch_size=mini_batch_size)

ae_model.kmeans_conclude()
vstream.reset_stream()
print "############################"
print "END_OF_EPOCH K MEANS"
print "############################"


ae_model.change_clustering_weight(1.0)
print "############################"
print "START_OF_EPOCHS TRAINING"
print "############################"

for i in range(0,n_epochs):
    print "############################"
    print "START_OF_EPOCHS TRAINING: ",(i+1)
    print "############################"
    for k in tqdm(range(0, length_run)):
        bf.process_a_batch(ae_model, vstream, thresh_variance=0.1, mini_batch_size=mini_batch_size, train=True)

    vstream.reset_stream()
    ae_model.save_weights()
    print "############################"
    print "END_OF_EPOCHS TRAINING: ",(i+1)
    print "############################"

    bf.do_recon(ae_model, vstream, thresh_variance=0.1, num_recons=10)

ae_model.generate_loss_graph('loss_graph.png')
ae_model.generate_mean_displacement_graph('mean_displacement_graph.png')

