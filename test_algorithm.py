import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from data_pkg.data_fns import TestDictionary,TrainDictionary,TestVideoStream
from sys import argv
import os
import socket
import h5py

metric = af.getopts(argv)

rdict = af.parse_run_variables(metric,set_mem=True)

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

tvs.set_GMMThreshold(threshold=25.0)

tvs.process_data()