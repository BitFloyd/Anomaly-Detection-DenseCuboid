import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from data_pkg.data_fns import TestDictionary
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
    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    #
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # set_session(tf.Session(config=config))

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
    n_gpus = int(metric['-ngpu'])

else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/densecub')
    path_videos = '/gs/project/suu-621-aa/sejacob/data/art_videos_prob_0.01/artif_videos_128x128'
    data_store_suffix = '/gs/scratch/sejacob/densecub'
    n_gpus = int(metric['-ngpu'])


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
    data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test','data_store_greyscale_test_bkgsub' + str(tstrides))
    nc=1

else:

    folder = os.path.join(data_store_suffix,'chapter_store_conv', 'data_store_bksgub' + str(tstrides) + '_0.0')
    data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test', 'data_store_test_bkgsub' + str(tstrides))
    nc=3

train_dset = h5py.File(os.path.join(folder,'data_train.h5'),'r')

if(n_chapters == 0):
    n_chapters=len(train_dset.keys())

if(nic==0):
    nic=n_chapters

train_dset.close()

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

if('-udiw' in metric.keys()):
    udiw = bool(int(metric['-udiw']))
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "USE DIST IN WORD CREATION :" , udiw
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

else:
    udiw = False

if('-tlm' in metric.keys()):
    tlm = metric['-tlm']
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "USE TEST LOSS METRIC:" , tlm
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

else:
    tlm = 'dssim'


suffix += '_large_'+str(large)
suffix +='_ntrain_'+str(ntrain)
suffix +='_lamda_'+str(lamda)
suffix +='_lassign_'+str(lassign)


notest = False

# Get MODEL
model_store = 'models/' + suffix

print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
print folder
print data_store
print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

if(notest):
    ae_model = None
else:
    ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=24, size_x=24, n_channels=3, h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters, clustering_lr=1,
                                            lr_model=1e-4, lamda=lamda, lamda_assign=lassign, n_gpus=1,gs=gs,notrain=True,
                                            reverse=False, data_folder=folder,dat_h5=None,large=large)

    ae_model.set_cl_loss(0.0)

#Get Test class
data_h5_vc = h5py.File(os.path.join(data_store,'data_test_video_cuboids.h5'))
data_h5_va = h5py.File(os.path.join(data_store,'data_test_video_anomgt.h5'))
data_h5_vp = h5py.File(os.path.join(data_store,'data_test_video_pixmap.h5'))

tclass = TestDictionary(ae_model,data_store=data_store,data_test_h5=[data_h5_vc,data_h5_va,data_h5_vp],
                        notest=notest,model_store=model_store,test_loss_metric=tlm,use_dist_in_word=udiw)

print "############################"
print "UPDATING DICT FROM DATA"
print "############################"
tclass.update_dict_from_data()


print "############################"
print "PRINTING DEETS AND PLOT FREQUENCY"
print "############################"
tclass.print_details_and_plot('word_frequency_udiw_'+str(udiw)+'.png',tlm+'_recon_losses.png')

print "############################"
print "MAKE LIST OF FULL CUBOID DATASET FREQUENCIES"
print "############################"
tclass.make_list_full_dset_cuboid_frequencies()

print "############################"
print "MAKE PRF CURVE FRQ"
print "############################"
tclass.make_p_r_f_a_curve('prf_curve_udiw_'+str(udiw)+'.png','tpfp_curve_udiw_'+str(udiw)+'.png','prf_deets_udiw_'+str(udiw)+'.txt')

print "############################"
print "MAKE PRF CURVE LOSS"
print "############################"
tclass.make_p_r_f_a_curve_dss('prf_curve_'+tlm+'.png','tpfp_curve_'+tlm+'.png','prf_deets_'+tlm+'.txt')

print "############################"
print "MAKE FREQUENCY SAMPLES PLOT"
print "############################"
tclass.plot_frequencies_of_samples('frequency_samples_plot_'+str(udiw)+'.png')

print "############################"
print "MAKE LOSS SAMPLES PLOT"
print "############################"
tclass.plot_loss_of_samples('loss_'+tlm+'_samples_plot.png')

print "############################"
print "MAKE DISTANCE METRIC PLOT"
print "############################"
tclass.plot_distance_measure_of_samples('distance_measure_mean_samples_plot.png','mean')
tclass.plot_distance_measure_of_samples('distance_measure_meanxloss_'+tlm+'_samples_plot.png','meanxloss')

print "############################"
print "MAKE DISTANCE METRIC PLOT"
print "############################"
tclass.plot_distance_measure_of_samples('distance_measure_std_samples_plot.png','std')
tclass.plot_distance_measure_of_samples('distance_measure_stdxloss_'+tlm+'_samples_plot.png','stdxloss')


print "############################"
print "MAKE NORM ANOM CUBOID PDFS"
print "############################"
tclass.create_distance_metric_pdfs(1000)