import matplotlib as mpl

mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from data_pkg.data_fns import TestDictionary, TrainDictionary
from sys import argv
import os
import socket
import h5py

metric = af.getopts(argv)

n_gpus = 1
server = 0

guill = False

if (socket.gethostname() == 'puck'):
    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"
    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    #
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.25
    # set_session(tf.Session(config=config))
    data_store_suffix = '/usr/local/data/sejacob/ANOMALY/densecub'

elif ('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/scratch/suu-621-aa/ANOMALY/densecub')
    data_store_suffix = '/scratch/suu-621-aa/ANOMALY/densecub'

else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/densecub')
    data_store_suffix = '/gs/scratch/sejacob/densecub'
    guill = True

    if ('-ngpu' in metric.keys()):
        n_gpus = int(metric['-ngpu'])

if (guill and '-ngpu' in metric.keys()):
    batch_size = 256 * n_gpus
else:
    batch_size = 256

n_chapters = 0
gs = False
nic = 0
tstrides = 4
lassign = 0.0
h_units = 128
loss = 'dssim'
ntrain = 1
nclusters = 10
nocl = True

if (nocl):
    suffix = 'nocl_tstrd_' + str(tstrides) + '_nic_' + str(nic) + '_chapters_' + str(n_chapters) + '_clusters_' + str(
        nclusters)
    lamda = 0.0
else:
    suffix = 'tstrd_' + str(tstrides) + '_nic_' + str(nic) + '_chapters_' + str(n_chapters) + '_clusters_' + str(
        nclusters)
    lamda = float(metric['-lamda'])

suffix += '_hunits_' + str(h_units)

if (gs):

    train_folder = os.path.join(data_store_suffix, 'chapter_store_conv',
                                'triangle_data_store_greyscale_bkgsub' + str(tstrides))
    test_data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test',
                                   'triangle_data_store_greyscale_test_bkgsub' + str(tstrides))
    nc = 1

else:

    train_folder = os.path.join(data_store_suffix, 'chapter_store_conv', 'triangle_data_store_bksgub' + str(tstrides))
    test_data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test',
                                   'triangle_data_store_test_bkgsub' + str(tstrides))
    nc = 3

# Open train dset
train_dset = h5py.File(os.path.join(train_folder, 'data_train.h5'), 'r')

if (n_chapters == 0):
    n_chapters = len(train_dset.keys())

if (nic == 0):
    nic = n_chapters

suffix += '_test_'

if (gs):
    suffix += '_greyscale_'
else:
    suffix += '_color_'

print "############################"
print "SET UP MODEL"
print "############################"

if ('-large' in metric.keys()):
    large = bool(int(metric['-large']))
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "MODEL SIZE LARGE? :", large
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

else:
    large = True

if ('-tlm' in metric.keys()):
    tlm = metric['-tlm']
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "USE TEST LOSS METRIC:", tlm
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

else:
    tlm = 'dssim'

if ('-rev' in metric.keys()):

    reverse = bool(int(metric['-rev']))
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "REVERSE? :", reverse
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    suffix += '_rev_' + str(reverse)

else:
    reverse = False

suffix += '_large_' + str(large)
suffix += '_ntrain_' + str(ntrain)
suffix += '_lamda_' + str(lamda)

# Get MODEL
model_store = 'models/' + suffix

size = 24

print "#############################"
print "LOAD MODEL"
print "#############################"

notrain = False

ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=size, size_x=size, n_channels=3,
                                            h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                            lr_model=1e-3, lamda=lamda, gs=gs, notrain=notrain,
                                            reverse=reverse, data_folder=train_folder, dat_h5=train_dset, large=large)

print "############################"
print "START TRAINING AND STUFF"
print "############################"

if (guill and '-ngpu' in metric.keys()):
    print "############################"
    print "TRYING TO PARALLELISE TO MULTIPLE GPUS"
    print "############################"
    ae_model.make_ae_model_multi_gpu(n_gpus)

if (nocl):
    ae_model.fit_model_ae_chaps_nocloss(verbose=1, earlystopping=True, patience=100, n_chapters=n_chapters,
                                        n_train=ntrain, reduce_lr=True, patience_lr=25, factor=1.25)


else:
    ae_model.fit_model_ae_chaps(verbose=1, n_initial_chapters=nic, earlystopping=True, patience=100,
                                n_chapters=n_chapters,
                                n_train=ntrain, reduce_lr=True, patience_lr=25, factor=1.25)

    ae_model.generate_mean_displacement_graph('mean_displacements.png')


ae_model.prepare_for_model_sanity()

print "#############################"
print "RELOAD MODEL FOR TEST"
print "#############################"

del ae_model
notrain = True

ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=size, size_x=size, n_channels=3,
                                            h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                            lr_model=1e-3, lamda=lamda, gs=gs, notrain=notrain,
                                            reverse=reverse, data_folder=train_folder, dat_h5=train_dset, large=large)

ae_model.check_model_sanity()

ae_model.perform_kmeans(n_chapters=n_chapters, partial=True)
ae_model.perform_dict_learn(n_chapters=n_chapters, guill=guill)
ae_model.generate_loss_graph('loss_graph.png')
ae_model.create_recons(20)
ae_model.mean_and_samples(n_per_mean=8)
ae_model.generate_assignment_graph('assignment_graph.png', n_chapters=n_chapters)
ae_model.decode_means('means_decoded')
ae_model.save_gifs_per_cluster_ids(n_samples_per_id=100, total_chaps_trained_on=n_chapters, max_try=10)

train_dset.close()


