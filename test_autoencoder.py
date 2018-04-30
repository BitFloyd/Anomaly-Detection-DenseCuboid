import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from data_pkg.data_fns import TestDictionary
from sys import argv
import os
import socket
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
    config.gpu_options.per_process_gpu_memory_fraction = 0.65
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



batch_size=256
n_chapters = 0
gs = False
nic = 0
tstrides = 4
lassign = 0.0
h_units = int(metric['-h'])
loss = 'dssim'
ntrain = int(metric['-ntrain'])
nclusters = int(metric['-nclust'])
nocl = bool(int(metric['-nocl']))


if(nocl):
    suffix = 'nocl_tstrd_'+str(tstrides)+'_nic_'+str(nic)+'_chapters_'+str(n_chapters) + '_clusters_'+str(nclusters)
    lamda = 0.0
else:
    suffix = 'tstrd_' + str(tstrides) + '_nic_' + str(nic) + '_chapters_' + str(n_chapters) + '_clusters_' + str(nclusters)
    lamda = float(metric['-lamda'])

suffix +='_hunits_'+str(h_units)

if(gs):

    folder = os.path.join(data_store_suffix,'chapter_store_conv','data_store_greyscale_bkgsub'+str(tstrides))
    data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test','data_store_greyscale_test_bkgsub' + str(tstrides))
    nc=1

else:

    folder = os.path.join(data_store_suffix,'chapter_store_conv', 'data_store_bksgub' + str(tstrides) + '_0.0')
    data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test', 'data_store_test_bkgsub' + str(tstrides))
    nc=3


# Open train dset
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
    large = True

if('-tlm' in metric.keys()):
    tlm = metric['-tlm']
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "USE TEST LOSS METRIC:" , tlm
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

else:
    tlm = 'dssim'

if('-rev' in metric.keys()):

    reverse = bool(int(metric['-rev']))
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "REVERSE? :" , reverse
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    suffix += '_rev_' + str(reverse)

else:
    reverse = False


suffix += '_large_'+str(large)
suffix +='_ntrain_'+str(ntrain)
suffix +='_lamda_'+str(lamda)




# Get MODEL
model_store = 'models/' + suffix

size = 24

ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=size, size_x=size, n_channels=3, h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                            lr_model=1e-3, lamda=lamda, gs=gs,notrain=False,
                                            reverse=reverse, data_folder=folder,dat_h5=train_dset,large=large)


print "############################"
print "START TRAINING AND STUFF"
print "############################"

if(nocl):
    ae_model.fit_model_ae_chaps_nocloss(verbose=1, earlystopping=True, patience=100, n_chapters=n_chapters,
                                        n_train=ntrain, reduce_lr=True, patience_lr=25, factor=1.25)

    ae_model.perform_kmeans(n_chapters=n_chapters)
    ae_model.kmeans_partial_fit_displacement_plot()


else:
    ae_model.fit_model_ae_chaps(verbose=1,n_initial_chapters=nic,earlystopping=True,patience=100,n_chapters=n_chapters,
                            n_train=ntrain, reduce_lr = True, patience_lr=25 , factor=1.25)

    ae_model.generate_mean_displacement_graph('mean_displacements.png')
    ae_model.perform_kmeans(n_chapters=n_chapters)
    ae_model.kmeans_partial_fit_displacement_plot()


ae_model.generate_loss_graph('loss_graph.png')

ae_model.create_recons(20)

ae_model.mean_and_samples(n_per_mean=8)

ae_model.generate_assignment_graph('assignment_graph.png',n_chapters=n_chapters)

ae_model.decode_means('means_decoded')

ae_model.save_gifs_per_cluster_ids(n_samples_per_id=100,total_chaps_trained_on=n_chapters,max_try=10)

ae_model.perform_dict_learn(n_chapters=n_chapters)

del ae_model
train_dset.close()


print "################################"
print "START TESTING"
print "################################"

notest = False
udiw = False

print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
print folder
print data_store
print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

if(notest):
    ae_model = None
else:
    ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=24, size_x=24, n_channels=3, h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                            lr_model=1e-3, lamda=lamda,gs=gs,notrain=True,
                                            reverse=False, data_folder=folder,dat_h5=None,large=large)

    ae_model.set_cl_loss(0.0)

#Get Test class
data_h5_vc = h5py.File(os.path.join(data_store,'data_test_video_cuboids.h5'))
data_h5_va = h5py.File(os.path.join(data_store,'data_test_video_anomgt.h5'))
data_h5_vp = h5py.File(os.path.join(data_store,'data_test_video_pixmap.h5'))
data_h5_ap = h5py.File(os.path.join(data_store,'data_test_video_anomperc.h5'))

tclass = TestDictionary(ae_model,data_store=data_store,data_test_h5=[data_h5_vc,data_h5_va,data_h5_vp,data_h5_ap],
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
tclass.make_p_r_f_curve_word_frequency('prf_curve_udiw_'+str(udiw)+'.png',
                                       'tpfp_curve_udiw_'+str(udiw)+'.png',
                                       'prf_deets_udiw_'+str(udiw)+'.txt',
                                       'word_frequency')

print "############################"
print "MAKE PRF CURVE LOSS"
print "############################"
tclass.make_p_r_f_curve_loss_metric('prf_curve_'+tlm+'.png',
                                    'tpfp_curve_'+tlm+'.png',
                                    'prf_deets_'+tlm+'.txt',
                                    tlm+' loss')

print "############################"
print "MAKE FREQUENCY SAMPLES PLOT"
print "############################"
tclass.plot_frequencies_of_samples('word_frequency_plot_udiw'+str(udiw)+'.png')

print "############################"
print "MAKE LOSS SAMPLES PLOT"
print "############################"
tclass.plot_loss_of_samples('loss_'+tlm+'_samples_plot.png')

print "############################"
print "MAKE DISTANCE METRIC PLOT"
print "############################"
tclass.plot_distance_measure_of_samples('distance_mean_samples_plot.png','mean')
tclass.plot_distance_measure_of_samples('distance_meanxloss_'+tlm+'_samples_plot.png','meanxloss')
tclass.plot_distance_measure_of_samples('distance_target_samples_plot.png','distance')
tclass.plot_distance_measure_of_samples('distancexloss_'+tlm+'_samples_plot.png','distancexloss')
print "############################"
print "MAKE NORM ANOM CUBOID PDFS"
print "############################"
tclass.create_distance_metric_pdfs(1000)