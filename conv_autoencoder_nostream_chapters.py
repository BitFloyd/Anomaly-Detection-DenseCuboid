import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from sys import argv
import os
import socket
import sys
import re
import h5py

metric = af.getopts(argv)

n_gpus = 1
server=0

if(socket.gethostname()=='puck'):
    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/art_videos_prob_0.01/artif_videos_128x128'
    data_store_suffix = '/usr/local/data/sejacob/ANOMALY/densecub'

elif('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/scratch/suu-621-aa/ANOMALY/densecub')
    path_videos='/scratch/suu-621-aa/ANOMALY/data/art_videos_prob_0.01/artif_videos_128x128'
    data_store_suffix = '/scratch/suu-621-aa/ANOMALY/densecub'


else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/densecub')
    path_videos = '/gs/project/suu-621-aa/sejacob/data/art_videos_prob_0.01/artif_videos_128x128'
    data_store_suffix = '/gs/scratch/sejacob/densecub/'



h_units = int(metric['-h'])
batch_size=int(metric['-bs'])
n_chapters = int(metric['-nch'])
gs = bool(int(metric['-gs']))
nic = int(metric['-i'])
tstrides = int(metric['-tstrd'])
loss = metric['-loss']
ntrain = int(metric['-ntrain'])
nclusters = int(metric['-nclust'])
lamda = float(metric['-lamda'])
lassign = 0.0
nocl = bool(int(metric['-nocl']))

if(nocl):
    suffix = 'nocl_tstrd_'+str(tstrides)+'_nic_'+str(nic)+'_chapters_'+str(n_chapters) + '_clusters_'+str(nclusters)
else:
    suffix = 'tstrd_' + str(tstrides) + '_nic_' + str(nic) + '_chapters_' + str(n_chapters) + '_clusters_' + str(nclusters)

suffix +='_hunits_'+str(h_units)


if(gs):
    folder = os.path.join(data_store_suffix,'chapter_store_conv','data_store_greyscale_bkgsub'+str(tstrides))
    nc=1
else:
    folder = os.path.join(data_store_suffix,'chapter_store_conv', 'data_store_bksgub' + str(tstrides) + '_0.0')
    nc=3

train_dset = h5py.File(os.path.join(folder,'data_train.h5'),'r')

if(n_chapters == 0):
    n_chapters=len(train_dset.keys())

if(nic==0):
    nic=n_chapters

if(gs):
    suffix += '_greyscale_'
else:
    suffix += '_color_'

print "############################"
print "SET UP MODEL"
print "############################"



if('-large' in metric.keys()):
    large = bool(int(metric['-large']))
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "MODEL SIZE LARGE? :" , large
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

else:
    large = False


suffix += '_large_'+str(large)
suffix +='_ntrain_'+str(ntrain)
suffix +='_lamda_'+str(lamda)
suffix +='_lassign_'+str(lassign)

# Get MODEL
model_store = 'models/' + suffix

size = 24

ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=size, size_x=size, n_channels=3, h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters, clustering_lr=1,
                                            lr_model=1e-4, lamda=lamda, lamda_assign=lassign, n_gpus=1,gs=gs,notrain=False,
                                            reverse=False, data_folder=folder,dat_h5=train_dset,large=large)


print "############################"
print "START TRAINING AND STUFF"
print "############################"

if(nocl):
    ae_model.fit_model_ae_chaps_nocloss(verbose=1, earlystopping=True, patience=100, n_chapters=n_chapters,
                                        n_train=ntrain, reduce_lr=True, patience_lr=25, factor=1.25)


else:
    ae_model.fit_model_ae_chaps(verbose=1,n_initial_chapters=nic,earlystopping=True,patience=100,n_chapters=n_chapters,
                            n_train=ntrain, reduce_lr = True, patience_lr=25 , factor=1.25)


ae_model.perform_kmeans(n_chapters=n_chapters)

ae_model.kmeans_partial_fit_displacement_plot()

ae_model.generate_loss_graph('loss_graph.png')

ae_model.create_recons(20)

ae_model.mean_and_samples(n_per_mean=8)

ae_model.generate_assignment_graph('assignment_graph.png',n_chapters=10,total_chaps_trained=n_chapters)

ae_model.mean_displacement_distance()

ae_model.generate_mean_displacement_graph('mean_displacements.png')

ae_model.decode_means('means_decoded')

ae_model.save_gifs_per_cluster_ids(n_samples_per_id=100,total_chaps_trained_on=n_chapters,max_try=100)

ae_model.create_tsne_plot('tsne_plot.png',n_chapters=10,total_chaps_trained=n_chapters)

train_dset.close()