import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from data_pkg.data_fns import TestDictionary,TrainDictionary,TestVideoStream,TrainDataGenerator
from sys import argv
import os
import h5py

metric = af.getopts(argv)

rdict = af.parse_run_variables(metric,set_mem=True,set_mem_value=0.65,ucsd2=True,greyscale=True)

n_gpus = rdict['n_gpus']
guill = rdict['guill']
data_store_suffix = rdict['data_store_suffix']
batch_size=rdict['batch_size']
n_chapters = rdict['n_chapters']
gs = rdict['gs']
nic = rdict['nic']
tstrides = rdict['tstrides']
lassign = rdict['lassign']
h_units = rdict['h_units']
loss = rdict['loss']
ntrain = rdict['ntrain']
nclusters = rdict['nclusters']
nocl = rdict['nocl']
lamda = rdict['lamda']
train_folder = rdict['train_folder']
test_data_store = rdict['test_data_store']
nc=rdict['nc']
large = rdict['large']
tlm = rdict['tlm']
reverse = rdict['reverse']
model_store = 'ucsd2_'+rdict['model_store']
use_basis_dict = rdict['use_basis_dict']
path_to_videos_test = rdict['path_to_videos_test']
sp_strides = rdict['sp_strides']

size = 48
do_silhouette = True


train_dset = h5py.File(os.path.join(train_folder, 'data_train.h5'), 'r')

print "#############################"
print "LOAD MODEL"
print "#############################"

notrain = True

if(h_units>0):
    ae_model = models.Conv_autoencoder_nostream_UCSD_h(model_store=model_store, size_y=size, size_x=size, n_channels=1,h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                            lr_model=1e-3, lamda=lamda, gs=gs,notrain=notrain,
                                            reverse=reverse, data_folder=train_folder,dat_h5=train_dset,large=large)
else:
    ae_model = models.Conv_autoencoder_nostream_UCSD_noh(model_store=model_store, size_y=size, size_x=size, n_channels=1,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                            lr_model=1e-3, lamda=lamda, gs=gs,notrain=notrain,
                                            reverse=reverse, data_folder=train_folder,dat_h5=train_dset,large=large)

print "############################"
print "START TRAINING AND STUFF"
print "############################"

if(guill and '-ngpu' in metric.keys()):
    print "############################"
    print "TRYING TO PARALLELISE TO MULTIPLE GPUS"
    print "############################"
    ae_model.make_ae_model_multi_gpu(n_gpus)

if(nocl):
    ae_model.fit_model_ae_chaps_nocloss(verbose=1, earlystopping=True, patience=100, n_chapters=n_chapters,
                                        n_train=ntrain, reduce_lr=True, patience_lr=30, factor=1.25)


else:
    ae_model.fit_model_ae_chaps(verbose=1,n_initial_chapters=nic,earlystopping=True,patience=100,n_chapters=n_chapters,
                            n_train=ntrain, reduce_lr = True, patience_lr=30 , factor=1.25)
    ae_model.generate_mean_displacement_graph('mean_displacements.png')

if(do_silhouette):
    ae_model.perform_num_clusters_analysis()

ae_model.perform_kmeans(partial=True)
ae_model.perform_dict_learn(guill=guill)
ae_model.generate_loss_graph('loss_graph.png')
ae_model.perform_feature_space_analysis()
ae_model.create_recons(20)
ae_model.mean_and_samples(n_per_mean=8)
ae_model.generate_assignment_graph('assignment_graph.png')
ae_model.decode_means('means_decoded')
ae_model.save_gifs_per_cluster_ids(n_samples_per_id=100,total_chaps_trained_on=n_chapters,max_try=10)
ae_model.perform_gmm_training(guill=guill,n_comp=nclusters)
ae_model.create_tsne_plot(graph_name='tsne_plot.png')

train_dset.close()

print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
print "CREATE AND SAVE DICTIONARY OF WORD FREQUENCIES FROM TRAINING DATA"
print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

#Get Train cuboids
data_h5_vc = h5py.File(os.path.join(train_folder,'data_train_video_cuboids.h5'))

train_dict = TrainDictionary(ae_model,data_train_h5=data_h5_vc,model_store=model_store)

print "############################"
print "CREATING DICTIONARY FROM TRAINING DATA"
print "############################"
train_dict.update_dict_from_data()

print "############################"
print "PRINTING DEETS AND PLOT FREQUENCY"
print "############################"
train_dict.print_details_and_plot('word_frequency_in_dict.png')


data_h5_vc.close()
del TrainDictionary


del ae_model


print "################################"
print "START TESTING"
print "################################"

notest = False
udiw = False

print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
print train_folder
print test_data_store
print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

if(notest):
    ae_model = None
else:
    if (h_units > 0):
        ae_model = models.Conv_autoencoder_nostream_UCSD_h(model_store=model_store, size_y=size, size_x=size,
                                                           n_channels=1, h_units=h_units,
                                                           n_timesteps=8, loss=loss, batch_size=batch_size,
                                                           n_clusters=nclusters,
                                                           lr_model=1e-3, lamda=lamda, gs=gs, notrain=True,
                                                           reverse=reverse, data_folder=train_folder, dat_h5=train_dset,
                                                           large=large)
    else:
        ae_model = models.Conv_autoencoder_nostream_UCSD_noh(model_store=model_store, size_y=size, size_x=size,
                                                             n_channels=1,
                                                             n_timesteps=8, loss=loss, batch_size=batch_size,
                                                             n_clusters=nclusters,
                                                             lr_model=1e-3, lamda=lamda, gs=gs, notrain=True,
                                                             reverse=reverse, data_folder=train_folder,
                                                             dat_h5=train_dset, large=large)

    ae_model.set_cl_loss(0.0)

    if(do_silhouette):
        ae_model.set_clusters_to_optimum()

#Get Test class
data_h5_vc = h5py.File(os.path.join(test_data_store,'data_test_video_cuboids.h5'))
data_h5_va = h5py.File(os.path.join(test_data_store,'data_test_video_anomgt.h5'))
data_h5_vp = h5py.File(os.path.join(test_data_store,'data_test_video_pixmap.h5'))
data_h5_ap = h5py.File(os.path.join(test_data_store,'data_test_video_anomperc.h5'))


tclass = TestDictionary(ae_model,data_store=test_data_store,data_test_h5=[data_h5_vc,data_h5_va,data_h5_vp,data_h5_ap],
                        notest=notest,model_store=model_store,test_loss_metric=tlm,use_dist_in_word=udiw,
                        use_basis_dict=use_basis_dict)


print "############################"
print "UPDATING DICT FROM DATA"
print "############################"
tclass.process_data()


print "############################"
print "MAKE LIST OF FULL CUBOID DATASET FREQUENCIES"
print "############################"
tclass.make_list_full_dset_cuboid_frequencies()

print "############################"
print "MAKE FREQUENCY SAMPLES PLOT"
print "############################"
tclass.plot_frequencies_of_samples('word_frequency_plot_udiw'+str(udiw)+'.png')

print "############################"
print "MAKE LOSS SAMPLES PLOT"
print "############################"
tclass.plot_loss_of_samples('loss_'+tlm+'_samples_plot.png')


if(use_basis_dict):

    print "############################"
    print "MAKE BASIS_DICT_RECON SAMPLES PLOT"
    print "############################"
    tclass.plot_basis_dict_recon_measure_of_samples('basis_dict_recon_error_samples_plot.png')

print "############################"
print "MAKE DISTANCE METRIC PLOT"
print "############################"
tclass.plot_distance_measure_of_samples('distance_mean_samples_plot.png','mean')
tclass.plot_distance_measure_of_samples('distance_meanxloss_'+tlm+'_samples_plot.png','meanxloss')
tclass.plot_distance_measure_of_samples('distance_target_samples_plot.png','distance')
tclass.plot_distance_measure_of_samples('distancexloss_'+tlm+'_samples_plot.png','distancexloss')

print "########################################################"
print "PERFORM FEATURE ANALYSIS ON ANOMALY VS NORMAL FEATURES"
print "########################################################"
tclass.feature_analysis_normvsanom()

print "########################################################"
print "PERFORM GMM ANALYSIS ON ANOMALY VS NORMAL FEATURES"
print "########################################################"
score_dict = tclass.gmm_analysis()

data_h5_vc.close()
data_h5_va.close()
data_h5_vp.close()
data_h5_ap.close()

threshold = score_dict['max_f1_th']
print "########################################################"
print "THRESHOLD: ", threshold
print "########################################################"

ae_model.load_gmm_model()

tvs = TestVideoStream(PathToVideos=path_to_videos_test,
                      CubSizeY=size,
                      CubSizeX=size,
                      CubTimesteps=8,
                      ModelStore=model_store,
                      Encoder=ae_model.encoder,
                      GMM=ae_model.gm,
                      LSTM=False,
                      StridesTime=tstrides,
                      StridesSpace=sp_strides,
                      GrayScale=gs,
                      BkgSub=True)

tvs.set_GMMThreshold(threshold=threshold)

tvs.process_data()