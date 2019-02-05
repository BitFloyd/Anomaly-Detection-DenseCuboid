import matplotlib as mpl
mpl.use('Agg')
from functionals_pkg import argparse_fns as af
import os
from sys import argv
from skimage.io import imread, imsave
import numpy as np
from tqdm import tqdm
import time

def strip_sth(list_to_be_stripped, strip_tag,strip_if_present=True):
    list_relevant = []

    for i in range(0, len(list_to_be_stripped)):

        splitted = list_to_be_stripped[i].split('_')

        if(strip_if_present):
            if (splitted[-1] == strip_tag):
                continue
            else:
                list_relevant.append(list_to_be_stripped[i])
        else:
            if (splitted[-1] != strip_tag):
                continue
            else:
                list_relevant.append(list_to_be_stripped[i])

    return list_relevant

def make_file_list(loc_videos):

    list_dirs = os.listdir(loc_videos)
    list_dirs.sort()
    list_dirs = strip_sth(list_dirs, strip_tag='Store')
    list_dirs = strip_sth(list_dirs, strip_tag='gt')
    list_dirs = strip_sth(list_dirs,strip_tag='Videos')


    list_images_relevant_full_dset = []

    for idx, i in enumerate(list_dirs):
        list_fnames = os.listdir(os.path.join(loc_videos,i))
        list_fnames.sort()
        list_fnames = strip_sth(list_fnames, strip_tag='Store')

        for index, j in enumerate(list_fnames):
            filename = os.path.join(loc_videos,i,j)
            list_images_relevant_full_dset.append(filename)



    return list_images_relevant_full_dset


metric = af.getopts(argv)

dataset = metric['-dataset']
gs = bool(int(metric['-gs']))

print "gs:",gs

if (dataset=='TRIANGLE'):
    train_data_path = '/usr/local/data/sejacob/FINISHED_WORK/ANOMALY/data/art_videos_triangle/Train'
elif(dataset=='UCSD1'):
    train_data_path = '/usr/local/data/sejacob/FINISHED_WORK/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train'
elif(dataset=='UCSD2'):
    train_data_path = '/usr/local/data/sejacob/FINISHED_WORK/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'


list_images = make_file_list(loc_videos=train_data_path)


sample_image = imread(list_images[0],as_grey=gs)

print "###############################"
print "Max Min Read:"
print np.max(sample_image)
print np.min(sample_image)
print "###############################"

running_avg = np.zeros(sample_image.shape)


for idx,i in tqdm(enumerate(list_images)):
    running_avg = (imread(i,as_grey=gs)+idx*running_avg)/(idx+1)

running_avg = np.uint8(running_avg)

print "###############################"
print "Max Min Save:"
print running_avg.shape
print np.max(running_avg)
print np.min(running_avg)
print "###############################"

save_filename = 'bkg_image'+dataset+'.'+list_images[0].split('.')[-1]
print save_filename
imsave(save_filename,running_avg)






