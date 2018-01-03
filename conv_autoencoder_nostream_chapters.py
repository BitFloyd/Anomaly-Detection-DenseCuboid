import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from sys import argv
import functionals_pkg.config as cfg
import os
import socket



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
    # config.gpu_options.per_process_gpu_memory_fraction = 0.75
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
    os.chdir('/gs/scratch/sejacob/densecub')
    path_videos = '/gs/project/suu-621-aa/sejacob/data/art_videos_prob_0.01/artif_videos_128x128'
    n_gpus = int(metric['-ngpu'])


h_units = int(metric['-h'])
batch_size=int(metric['-bs'])
n_chapters = int(metric['-nch'])
gs = bool(int(metric['-gs']))
nic = int(metric['-i'])
tstrides = int(metric['-tstrd'])
loss = metric['-loss']
ntrain = int(metric['-ntrain'])

suffix = 'tstrd_'+str(tstrides)+'_nic_'+str(nic)+'_chapters_'+str(n_chapters)

if(gs):
    folder = os.path.join('chapter_store_conv','data_store_greyscale_'+str(tstrides))
    nc=1
else:
    folder = os.path.join('chapter_store_conv','data_store_'+str(tstrides))
    nc=3

if(n_chapters == 0):
    n_chapters=len(os.listdir(folder))-2
if(nic==0):
    nic=n_chapters
    suffix='tstrd_'+str(tstrides)+'_recon_only_h_units'+str(h_units)+'_'+str(loss)

print "############################"
print "SET UP MODEL"
print "############################"

# Get MODEL
model_store = 'models/conv_autoencoder_chapters_'+suffix

ae_model = models.Conv_autoencoder_nostream(model_store=model_store,loss=loss,h_units=h_units,n_timesteps=5,n_channels=nc,
                                        batch_size=batch_size,n_clusters=10, clustering_lr=1, lr_model=1e-5,lamda=lamda,
                                        lamda_assign=lamda_assign,n_gpus=n_gpus,gs=gs,notrain=False,data_folder=folder,reverse=False)

print "############################"
print "START TRAINING AND STUFF"
print "############################"

for _ in range(0,ntrain):
    ae_model.fit_model_ae_chaps(verbose=1,n_initial_chapters=nic,earlystopping=True,patience=10,n_chapters=n_chapters)

ae_model.generate_loss_graph('loss_graph.png')

ae_model.create_recons(20)

if(nic<n_chapters):
    ae_model.generate_assignment_graph('assignment_graph.png',n_chapters=5,total_chaps_trained=n_chapters)

    ae_model.mean_displacement_distance()

    ae_model.generate_mean_displacement_graph('mean_displacements.png')

    ae_model.decode_means('means_decoded.png')

    ae_model.create_tsne_plot('tsne_plot.png',n_chapters=5,total_chaps_trained=n_chapters)

    ae_model.create_tsne_plot3d('tsne_plot3d.png',n_chapters=n_chapters,total_chaps_trained=n_chapters)