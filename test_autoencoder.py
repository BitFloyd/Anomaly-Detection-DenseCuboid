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
# lassign = float (metric['-lassign'])
lassign = 0.0
nocl = bool(int(metric['-nocl']))

if(nocl):
    suffix = 'nocl_tstrd_'+str(tstrides)+'_nic_'+str(nic)+'_chapters_'+str(n_chapters) + '_clusters_'+str(nclusters)
else:
    suffix = 'tstrd_' + str(tstrides) + '_nic_' + str(nic) + '_chapters_' + str(n_chapters) + '_clusters_' + str(nclusters)

suffix +='_hunits_'+str(h_units)

if(gs):
    if('-bkgsub' in metric.keys()):
        folder = os.path.join('chapter_store_conv','data_store_greyscale_bkgsub'+str(tstrides))
        data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test',
                                  'data_store_greyscale_test_bkgsub' + str(tstrides))
    else:
        folder = os.path.join('chapter_store_conv', 'data_store_greyscale_' + str(tstrides))
        data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test',
                                  'data_store_greyscale_test' + str(tstrides))
    nc=1
else:
    if('-bkgsub' in metric.keys()):
        folder = os.path.join('chapter_store_conv', 'data_store_bkgsub' + str(tstrides) + '_0.0')
        data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test', 'data_store_test_bkgsub' + str(tstrides))
    else:
        folder = os.path.join('chapter_store_conv', 'data_store_' + str(tstrides) + '_0.0')
        data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test',
                                  'data_store_test' + str(tstrides))
    nc=3

if(n_chapters == 0):
    r = re.compile('chapter_.*.npy')
    n_chapters=len(filter(r.match,os.listdir(folder)))

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
                                            reverse=False, data_folder=folder,large=large)

    ae_model.set_cl_loss(0.0)

#Get Test class
tclass = TestDictionary(ae_model,data_store=data_store,notest=notest,model_store=model_store,test_loss_metric='mse')

print "############################"
print "UPDATING DICT FROM DATA"
print "############################"
tclass.update_dict_from_data()


print "############################"
print "PRINTING DEETS AND PLOT FREQUENCY"
print "############################"
tclass.print_details_and_plot('word_frequency.png','dssim_recon_losses.png')

print "############################"
print "MAKE LIST OF FULL CUBOID DATASET FREQUENCIES"
print "############################"
tclass.make_list_full_dset_cuboid_frequencies()

print "############################"
print "MAKE PRF CURVE FRQ"
print "############################"
tclass.make_p_r_f_a_curve('prf_curve.png','tpfp_curve.png','prf_deets.txt')

print "############################"
print "MAKE PRF CURVE LOSS"
print "############################"
tclass.make_p_r_f_a_curve_dss('prf_curve_dss.png','tpfp_curve_dss.png','prf_deets_dss.txt')

print "############################"
print "MAKE FREQUENCY SAMPLES PLOT"
print "############################"
tclass.plot_frequencies_of_samples('frequency_samples_plot.png')

# print "############################"
# print "MAKE ANOMALY GIFS"
# print "############################"
# tclass.make_anomaly_gifs()