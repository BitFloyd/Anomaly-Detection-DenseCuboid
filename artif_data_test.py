import os
import socket

if(socket.gethostname()=='puck'):

    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))
elif('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/scratch/suu-621-aa/ANOMALY/densecub')
else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/densecub/')

import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from sys import argv



metric = af.getopts(argv)
print metric

model_store = 'models/artificial_data'

adt = models.Artificial_Data_Test(model_store=model_store,n_samples_each = 50000, proj_dim = 128, loss='mse',
                                  batch_size=128, clustering_lr=1, lr_model=1e-5, lamda=1e-3, assignment_lamda = 1e-3)

adt.plot_data_2d('data_2d_points.png')

adt.fit_model_ae(verbose=1,n_initial_epochs=int(metric['-i']),n_train_epochs=int(metric['-t']),earlystopping=True,patience=10,least_loss=1e-5)

adt.generate_loss_graph('loss_graph.png')

adt.generate_assignment_graph('assignment_graph.png')

adt.create_2d_encoding_plot('2d_encoding_plot.png')

adt.mean_displacement_distance()

adt.generate_mean_displacement_graph('mean_displacements.png')

