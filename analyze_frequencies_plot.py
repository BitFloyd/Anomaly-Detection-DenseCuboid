import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from matplotlib.colors import ListedColormap
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from data_pkg.data_fns import TestDictionary
from sys import argv
import os
import socket
import sys
import re
import time

metric = af.getopts(argv)

n_gpus = 1
server=0

x = np.random.rand(24,24,24)


def onclick(event):
    if(event.inaxes is not None):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  (event.button, event.x, event.y, event.xdata, event.ydata))
        ax = event.inaxes
        print "CLICK DETECTED ON AXES:", ax.name
        global x
        if(ax.name=='frequency_both'):
            x = np.load(os.path.join('cuboids','all_cubs',str(args_frq_sort[int(event.xdata)])+'.npy'))
        elif(ax.name=='frequency_anom'):
            x = np.load(os.path.join('cuboids', 'anom_cubs', str(args_frq_sort_gt[int(event.xdata)]) + '.npy'))
        elif(ax.name=='loss_both'):
            x = np.load(os.path.join('cuboids', 'all_cubs', str(args_loss_sort[int(event.xdata)]) + '.npy'))
        elif(ax.name=='loss_anom'):
            x = np.load(os.path.join('cuboids', 'anom_cubs', str(args_loss_sort_gt[int(event.xdata)]) + '.npy'))
        else:
            print "INVALID AXES NAME"
            sys.exit(0)
        
        print "FINISH CLICK EVENT"

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
lassign = float (metric['-lassign'])
nocl = bool(int(metric['-nocl']))

if(nocl):
    suffix = 'nocl_tstrd_'+str(tstrides)+'_nic_'+str(nic)+'_chapters_'+str(n_chapters) + '_clusters_'+str(nclusters)
else:
    suffix = 'tstrd_' + str(tstrides) + '_nic_' + str(nic) + '_chapters_' + str(n_chapters) + '_clusters_' + str(nclusters)

suffix +='_hunits_'+str(h_units)

if(gs):
    folder = os.path.join('chapter_store_conv','data_store_greyscale_'+str(tstrides))
    data_store = os.path.join(data_store_suffix,'chapter_store_conv_test','data_store_greyscale_test'+str(tstrides))
    nc=1
else:
    folder = os.path.join('chapter_store_conv','data_store_' + str(tstrides) + '_0.0')
    data_store = os.path.join(data_store_suffix,'chapter_store_conv_test','data_store_test'+str(tstrides))
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


notest = True

# Get MODEL
model_store = 'models/' + suffix


if(notest):
    ae_model = None
else:
    ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=24, size_x=24, n_channels=3, h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters, clustering_lr=1,
                                            lr_model=1e-4, lamda=lamda, lamda_assign=lassign, n_gpus=1,gs=gs,notrain=True,
                                            reverse=False, data_folder=folder,large=large)

    ae_model.set_cl_loss(0.0)

#Get Test class
tclass = TestDictionary(ae_model,data_store=data_store,notest=notest,model_store=model_store)

print "############################"
print "SET UP TCLASS"
print "############################"

y_true = np.array(tclass.list_of_cub_anom_gt_full_dataset)
frequency_array = np.array(tclass.list_full_dset_cuboid_frequencies)
loss_array = np.array(tclass.list_of_cub_loss_full_dataset)

fq_anoms =frequency_array[y_true==1]
args_frq_sort = np.argsort(frequency_array)
y_true_frq = y_true[args_frq_sort]
args_frq_sort_gt = np.argsort(fq_anoms)


frequency_array = frequency_array[args_frq_sort]
fq_anoms = fq_anoms[args_frq_sort_gt]

colors = ['green', 'red']

print "############################"
print "START PLOTTING"
print "############################"

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(50,40))


im1 = ax1.scatter(range(0, len(frequency_array)), frequency_array, c=y_true_frq, cmap=ListedColormap(colors), alpha=0.5 ,s=10)
ax1.set_title('ANOMS:Red, N-ANOMS:Green')
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Cuboid index')
ax1.name = 'frequency_both'
ax1.grid(True)
cb1 = f.colorbar(im1, ax=ax1)
loc = np.arange(0, max(y_true), max(y_true) / float(len(colors)))
cb1.set_ticks(loc)
cb1.set_ticklabels(['normal', 'anomaly'])


im2 = ax2.scatter(range(0, len(fq_anoms)), fq_anoms, c='red', alpha=0.5, s=10)
ax2.set_title('ANOMS:Red')
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Cuboid index')
ax2.name ='frequency_anom'
ax2.grid(True)

loss_anoms = loss_array[y_true== 1]
args_loss_sort = np.argsort(loss_array)
args_loss_sort_gt = np.argsort(loss_anoms)

y_true_loss = y_true[args_loss_sort]
loss_array = loss_array[args_loss_sort]
loss_anoms = loss_anoms[args_loss_sort_gt]

im3 = ax3.scatter(range(0, len(loss_array)), loss_array, c=y_true_loss, cmap=ListedColormap(colors), alpha=0.5,s=10)
ax3.set_title('ANOMS:Red, N-ANOMS:Green')
ax3.set_ylabel('DSSIM Loss')
ax3.set_xlabel('Cuboid index')
ax3.name='loss_both'
ax3.grid(True)
cb3 = f.colorbar(im3, ax=ax3)
loc = np.arange(0, max(y_true), max(y_true) / float(len(colors)))
cb3.set_ticks(loc)
cb3.set_ticklabels(['normal', 'anomaly'])


im4 = ax4.scatter(range(0, len(loss_anoms)), loss_anoms, c='red', alpha=0.5,s=10)
ax4.set_title('ANOMS:Red')
ax4.set_ylabel('DSSIM Loss')
ax4.set_xlabel('Cuboid index')
ax4.name='loss_anom'
ax4.grid(True)

f_new, ax_new = plt.subplots(1)

def update(i):
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    ax_new.clear()
    ax_new.imshow(i)

    # print "RUN UPDATE"
    return [ax_new]

def gen_frames():
    global x
    for i in range(0,x.shape[-1]/3):
        # print "GENERATING FRAME:",i
        yield x[:,:,i*3:(i+1)*3]

anim = FuncAnimation(f_new, update, frames=gen_frames, interval=200,repeat=True,blit=True)
cid = f.canvas.mpl_connect('button_press_event', onclick)


plt.show()


