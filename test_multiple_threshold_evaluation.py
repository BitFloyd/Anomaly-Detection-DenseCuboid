import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from data_pkg.data_fns import TestDictionary,TestVideoStream
from sys import argv
import os
import h5py

metric = af.getopts(argv)

rdict = af.parse_run_variables(metric,set_mem=True,set_mem_value=0.65)

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
size  = rdict['size']
min_data_threshold = rdict['min_data_threshold']
patience = rdict['patience']
bkgsub = rdict['bkgsub']

do_silhouette = True

train_dset = h5py.File(os.path.join(train_folder, 'data_train.h5'), 'r')


print "#############################"
print "LOAD MODEL"
print "#############################"

notrain = False

ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=size, size_x=size, n_channels=nc, h_units=h_units,
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

ae_model.fit_model_ae_chaps(verbose=1,n_initial_chapters=nic,earlystopping=True,patience=patience,n_chapters=n_chapters,
                                n_train=ntrain, reduce_lr = True, patience_lr=int(patience*0.66),
                                factor=1.25)


ae_model.perform_kmeans(partial=True)



list_thresholds = []

for i in range (0,15):
    notrain=True

    ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=size, size_x=size, n_channels=nc,
                                                h_units=h_units,
                                                n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                                lr_model=1e-3, lamda=lamda, gs=gs, notrain=notrain,
                                                reverse=reverse, data_folder=train_folder, dat_h5=train_dset,
                                                large=large)

    ae_model.perform_gmm_training(guill=guill,n_comp=nclusters,n_init=1)


    print "################################"
    print "START TESTING"
    print "################################"

    notest = False
    udiw = False

    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print train_folder
    print test_data_store
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

    ae_model.set_cl_loss(0.0)

    #Get Test class
    data_h5_vc = h5py.File(os.path.join(test_data_store,'data_test_video_cuboids.h5'))
    data_h5_va = h5py.File(os.path.join(test_data_store,'data_test_video_anomgt.h5'))
    data_h5_vp = h5py.File(os.path.join(test_data_store,'data_test_video_pixmap.h5'))
    data_h5_ap = h5py.File(os.path.join(test_data_store,'data_test_video_anomperc.h5'))


    tclass = TestDictionary(ae_model,data_store=test_data_store,data_test_h5=[data_h5_vc,data_h5_va,data_h5_vp,data_h5_ap],
                            notest=notest,model_store=model_store,test_loss_metric=tlm,use_dist_in_word=udiw,
                            use_basis_dict=use_basis_dict)


    print "########################################################"
    print "PERFORM GMM_ANALYSIS"
    print "########################################################"
    score_dict = tclass.gmm_analysis()

    data_h5_vc.close()
    data_h5_va.close()
    data_h5_vp.close()
    data_h5_ap.close()

    threshold = score_dict['max_f1_th']
    list_thresholds.append(threshold)



fig, ax = plt.subplots(figsize =(10,10))

y = list_thresholds
x = range(0,len(list_thresholds))

plt.ylabel('THRESHOLD FOR MAXIMUM F1-SCORE')
plt.xlabel('RUN-INDEX')
plt.title('THRESHOLDS FOR EACH RUN OF EM AND EVALUATION ON THE SAME ENCODER')
(markerLines, stemLines, baseLines) = plt.stem(x,y,linefmt='-.')
plt.setp(markerLines,markersize = 15)
plt.savefig('multiple_threshold_runs.png')
plt.show()
