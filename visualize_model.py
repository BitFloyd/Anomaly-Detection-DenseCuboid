import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from keras.utils import plot_model
from sys import argv
import socket
import os



metric = af.getopts(argv)

if (socket.gethostname() == 'puck'):
    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"
    data_store_suffix = '/usr/local/data/sejacob/ANOMALY/densecub'
    use_basis_dict = False

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

batch_size=256
h_units = 32
loss = 'dssim'
nclusters = 10
lamda = 0.01
train_folder = None
large = True
reverse = False
model_store = os.path.join(os.getcwd(),'plot_images')
gs = True

size = 24

do_silhouette = True

train_dset = None

print "#############################"
print "LOAD MODEL"
print "#############################"

notrain = False

ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=size, size_x=size, n_channels=3, h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                            lr_model=1e-3, lamda=lamda, gs=gs,notrain=notrain,
                                            reverse=reverse, data_folder=train_folder,dat_h5=train_dset,large=large)


print "#############################"
print "ENCODER MODEL:"
print "#############################"
ae_model.encoder.summary()
print "#############################"
print "DECODER MODEL:"
print "#############################"
ae_model.decoder.summary()
print "#############################"


plot_model(ae_model.ae,to_file=os.path.join(model_store,'ae_model.pdf'),show_layer_names=False)
plot_model(ae_model.encoder,to_file=os.path.join(model_store,'encoder.pdf'),show_layer_names=False)
plot_model(ae_model.decoder,to_file=os.path.join(model_store,'decoder.pdf'),show_layer_names=False)