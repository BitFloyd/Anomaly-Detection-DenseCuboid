import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from data_pkg.data_fns import TestDictionary,TrainDictionary,TestVideoStream,TrainDataGenerator
from sys import argv
import os
import h5py

metric = af.getopts(argv)

rdict = af.parse_run_variables(metric,set_mem=True,set_mem_value=0.5)

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
model_store = rdict['model_store']
use_basis_dict = rdict['use_basis_dict']
path_to_videos_test = rdict['path_to_videos_test']
sp_strides = rdict['sp_strides']

size = 24

do_silhouette = True

print "################################"
print "START TESTING"
print "################################"

notest = True
udiw = False

print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
print train_folder
print test_data_store
print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

if(notest):
    ae_model = None
else:
    ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=24, size_x=24, n_channels=3, h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                            lr_model=1e-3, lamda=lamda,gs=gs,notrain=True,
                                            reverse=reverse, data_folder=train_folder,dat_h5=None,large=large)

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


data_h5_vc.close()
data_h5_va.close()
data_h5_vp.close()
data_h5_ap.close()


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
    ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=24, size_x=24, n_channels=3, h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                            lr_model=1e-3, lamda=lamda,gs=gs,notrain=True,
                                            reverse=reverse, data_folder=train_folder,dat_h5=None,large=large)

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

score_dict = tclass.gmm_analysis()

data_h5_vc.close()
data_h5_va.close()
data_h5_vp.close()
data_h5_ap.close()

threshold = score_dict['max_f1_th']

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