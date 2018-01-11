import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from sys import argv
import functionals_pkg.config as cfg
import os
import socket
import sys


lamda = cfg.lamda
lamda_assign = cfg.lamda_assign

metric = af.getopts(argv)

n_gpus = 1
server=0

if(socket.gethostname()=='puck'):
    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"
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
    folder = os.path.join('chapter_store_lstm','data_store_greyscale_'+str(tstrides))
    nc=1
else:
    folder = os.path.join('chapter_store_lstm','data_store_'+str(tstrides))
    nc=3

if(n_chapters == 0):
    n_chapters=len(os.listdir(folder))-2
if(nic==0):
    nic=n_chapters
    suffix='tstrd_'+str(tstrides)+'_recon_only_h_units'+str(h_units)+'_'+str(loss)

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

if('-model' in metric.keys()):
    if str(metric['-model']) == 'bn':
        suffix += '_' + str(metric['-model'])
        # Get MODEL
        model_store = 'models/conv_lstmac_chapters_' + suffix
        ae_model = models.Conv_LSTM_autoencoder_nostream_nocl(model_store=model_store, loss=loss, h_units=h_units,
                                                              n_timesteps=5, n_channels=nc, batch_size=batch_size,
                                                              lr_model=1e-5, n_gpus=n_gpus, gs=gs, notrain=False,
                                                              data_folder=folder, reverse=False, large=large)

    elif str(metric['-model']) == 'nobn':
        suffix += '_' + str(metric['-model'])
        # Get MODEL
        model_store = 'models/conv_lstmac_chapters_' + suffix
        ae_model = models.Conv_LSTM_autoencoder_nostream_nocl(model_store=model_store, loss=loss, h_units=h_units,
                                                              n_timesteps=5, n_channels=nc, batch_size=batch_size,
                                                              lr_model=1e-5, n_gpus=n_gpus, gs=gs, notrain=False,
                                                              data_folder=folder, reverse=False, large=large)


    else :
        print "INVALID -MODEL PARAM, either use bn or nobn"
        sys.exit()
else:
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "USING NO BATCH-NORM MODEL"
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    suffix += '_' + 'nobn'
    # Get MODEL
    model_store = 'models/conv_lstmac_chapters_' + suffix
    ae_model = models.Conv_LSTM_autoencoder_nostream_nocl_nobn(model_store=model_store,loss=loss,h_units=h_units,n_timesteps=5,n_channels=nc,
                                                 batch_size=batch_size,lr_model=1e-5,n_gpus=n_gpus,gs=gs,notrain=False,data_folder=folder,reverse=False,large=large)


print "############################"
print "START TRAINING AND STUFF"
print "############################"
if(ntrain>0):
    for _ in range(0,ntrain):
        ae_model.fit_model_ae_chaps(verbose=1,n_initial_chapters=nic)

else:
    ae_model.fit_model_ae(verbose=1)

ae_model.generate_loss_graph('loss_graph.png')

ae_model.create_recons(20)

