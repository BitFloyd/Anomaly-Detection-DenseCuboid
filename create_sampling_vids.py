import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from functionals_pkg import argparse_fns as af
from functionals_pkg import batch_fns as bf
from sys import argv
from data_pkg import data_fns as df
import os
import socket
import numpy as np
import sys

metric = af.getopts(argv)

def plot_figures(figures, nrows = 1, ncols=1,save_name=None):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : array of images to be plotted
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=(20,20))

    for i in range(0,figures.shape[0]):
        for j in range(0,figures.shape[1]):
            axeslist[i][j].imshow(figures[i][j], cmap=plt.gray())
            axeslist[i][j].set_title('('+str(i)+','+str(j)+')')
            axeslist[i][j].set_axis_off()

    plt.tight_layout() # optional
    plt.savefig(save_name,bbox_inches='tight')
    plt.close()

def create_videos(frames_folder,folder_video_store,video_id):

    #Saves video and deletes everything in the frames_folder
    str_command = 'avconv -r 4 -y -i '+ os.path.join(frames_folder,'%10d.png') + ' ' + os.path.join(folder_video_store,str(video_id[0])+'.mp4')
    os.system(str_command)
    str_command = 'rm -rf '+frames_folder+'/*'
    os.system(str_command)

    return True

if (socket.gethostname() == 'puck'):
    print "############################################"
    print "DETECTED RUN ON PUCK"
    print "############################################"
    path_videos = '/usr/local/data/sejacob/ANOMALY/data/art_videos_prob_0.01/artif_videos_128x128'

elif ('gpu' in socket.gethostname()):
    print "############################################"
    print "DETECTED RUN ON HELIOS: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/scratch/suu-621-aa/ANOMALY/densecub')
    path_videos = '/scratch/suu-621-aa/ANOMALY/data/art_videos_prob_0.01/artif_videos_128x128'

else:
    print socket.gethostname()
    print "############################################"
    print "DETECTED RUN ON GUILLIMIN: Probably"
    print "############################################"
    verbose = 1
    os.chdir('/gs/project/suu-621-aa/sejacob/densecub')
    path_videos = '/gs/project/suu-621-aa/sejacob/data/art_videos_prob_0.01/artif_videos_128x128'


strides = int(metric['-strd'])
gs = bool(int(metric['-gs']))
tstrides = int(metric['-tstrd'])
lstm = bool(int(metric['-lstm']))

test = 1


mainfol = 'create_vids'

if(lstm):
    ts = 'first'
    ts_pos = 0


else:
    ts = 'last'
    ts_pos = -1


tv = 0.0

if(gs):
    folder = os.path.join(mainfol,'video_store_gs_bkgsub'+str(tstrides))


else:
    folder = os.path.join(mainfol,'video_store_bkgsub'+str(tstrides))



if(not os.path.exists(mainfol)):
    os.mkdir(mainfol)

if(not os.path.exists(folder)):
    os.mkdir(folder)

print "############################"
print "SET UP VIDEO STREAM"
print "############################"

#Get Data Stream
train_test = 'Test'

size_axis = 24
n_frames = 8
vstream = df.Video_Stream_ARTIF(video_path=path_videos, video_train_test=train_test, size_y=size_axis, size_x=size_axis,
                                timesteps=n_frames,ts_first_or_last=ts,strides=strides,tstrides=tstrides,anompth=0.1,
                                bkgsub=True)


print "############################"
print "ASSEMBLE AND SAVE DATASET"
print "############################"


if(os.path.exists(os.path.join(folder,'vid_this.npy'))):
    video_id = np.load(os.path.join(folder,'vid_this.npy'))


else:
    video_id = np.array([0])



print "SAVING FOR VIDEO ID:"
print video_id


vstream.set_seek_to_video(video_id[0])

print "SEEK IS NOW:", vstream.seek

video_frames_loc = os.path.join(folder,'frames')

if(not os.path.exists(video_frames_loc)):
    os.makedirs(video_frames_loc)

fid = 0

while True:

    anom_stats = False

    list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly = bf.return_cuboid_test(vstream,thresh_variance=tv,gs=gs,ts_pos=ts_pos)

    array_frames = list_cuboids[-1][:, :, :, :, -4:-1]

    plot_figures(array_frames,nrows = array_frames.shape[0],ncols=array_frames.shape[1],
                 save_name=os.path.join(video_frames_loc,str(fid).zfill(10)+'.png'))
    print fid
    fid += 1

    if(vstream.seek+1 == len(vstream.seek_dict.values())):
        break

    if (len(vstream.seek_dict[vstream.seek+1]) > 1):
        break


print "#########################################"
print "SAVING VIDEO: ",video_id
create_videos(video_frames_loc,folder,video_id)
print "#########################################"

video_id+=1
np.save(os.path.join(folder,'vid_this.npy'),video_id)




