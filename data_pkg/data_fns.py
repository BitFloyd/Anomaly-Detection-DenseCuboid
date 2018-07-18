import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import color,img_as_float
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import roc_curve,auc
from collections import deque
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from functionals_pkg import save_objects as so
from functionals_pkg.logging import debug_print,message_print
import seaborn as sns
import copy
import sys
import pickle
import math
import imageio
import h5py
import time
import shutil
import keras

sns.set(color_codes=True)

list_anom_percentage = []

class TrainDataGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, datafolder, batch_size=32, size=32, n_channels=1, n_timesteps = 8, shuffle=True):
        'Initialization'
        self.dim = (size,size,n_timesteps*n_channels)
        self.size = size
        self.n_timesteps = n_timesteps
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.list_datafiles = [os.path.join(datafolder,i) for i in os.listdir(datafolder)]
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_datafiles) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_datafiles[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)
        y = None
        return (X,y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_datafiles))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size,self.size,self.size,self.n_channels*self.n_timesteps))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(ID)

        return X

class TestImgClass:

    def __init__(self,fnames):

        self.fnames = fnames

    def get_time_base_filenames(self):
        fnames_past = self.fnames[0:-2]
        fnames_present = self.fnames[1:-1]
        fnames_future = self.fnames[2:]

        return fnames_past,fnames_present,fnames_future

class TrainDictionary:

    def __init__(self, model, data_train_h5=None, model_store=None):

        self.model = None
        self.dictionary_words = {}
        self.means = None

        self.data_store = None  # Where the train data is stored in spatiotemporal format

        self.vid = None  # Id of the video being processed ATM -> Always points to the next id to be loaded
        self.cubarray = None  # Array of all cuboids sampled in a video, arranged in a spatio-temporal fashion

        self.cubarray_process_index = None  # The index of the set in cubarray that is being processed.
        self.timesteps = 8
        self.data_vc = data_train_h5
        self.model_enc = model.encoder
        self.model_ae = model.ae
        self.model_store = model.model_store

        self.means = model.means  # means of each cluster learned by the model
        self.vid = 1
        self.model_store = model_store



    def load_data(self):

        # print os.path.join(self.data_store,'video_cuboids_array_'+str(self.vid)+'.npy')
        if ('video_cuboids_array_' + str(self.vid) in self.data_vc.keys()):
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$"
            print "LOADING VIDEO ARRAY ", self.vid
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            self.cubarray = np.array(self.data_vc.get('video_cuboids_array_' + str(self.vid)))
            print self.cubarray.shape

            self.vid += 1
            self.cubarray_process_index = 1
            return True

        else:
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$"
            print "VIDEOS FINISHED"
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            return False

    def save_h5data(self, name, content):

        with h5py.File(os.path.join(self.model_store, name + '.h5'), "a") as f:
            if (name in f.keys()):
                del f[name]

            dset = f.create_dataset(name, data=content)

        return True

    def load_h5data(self, name):
        dset = h5py.File(os.path.join(self.model_store, name + '.h5'), 'r')
        content = list(dset.get(name))
        dset.close()
        return content

    def fetch_next_set_from_cubarray(self):

        if (self.cubarray_process_index < len(self.cubarray) - 1):
            print self.cubarray_process_index, '/', len(self.cubarray) - 1
            sys.stdout.write("\033[F")
            three_rows_cubarray = self.cubarray[self.cubarray_process_index - 1:self.cubarray_process_index + 2]
            self.cubarray_process_index += 1

            return three_rows_cubarray,True
        else:
            return None,False

    def create_surroundings(self, three_rows_cubarray):

        rows = three_rows_cubarray[0].shape[0]
        cols = three_rows_cubarray[0].shape[1]

        list_surrounding_cuboids_from_three_rows = []

        for j in range(1, rows - 1):
            for k in range(1, cols - 1):
                surroundings = []

                surr_idx = 0

                current_cuboid = three_rows_cubarray[1][j, k]

                surroundings.append(current_cuboid)

                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j, k])
                surroundings.append(three_rows_cubarray[surr_idx][j, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k + 1])

                surr_idx = 1
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k + 1])

                surr_idx = 2
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j, k])
                surroundings.append(three_rows_cubarray[surr_idx][j, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k + 1])

                list_surrounding_cuboids_from_three_rows.append(np.array(surroundings))

        return np.array(list_surrounding_cuboids_from_three_rows)

    def predict_on_surroundings(self, surroundings_from_three_rows):

        list_predictions = []

        for i in range(0, len(surroundings_from_three_rows)):
            list_predictions.append(self.model_enc.predict(surroundings_from_three_rows[i]))

        return np.array(list_predictions)

    def create_words_from_predictions(self, predictions):

        list_words = []

        for i in range(0, len(predictions)):
            x = predictions[i]
            dist = cdist(x, self.means)
            word = tuple(np.argmin(dist, 1))
            list_words.append(word)

        return list_words

    def update_dict_with_words(self, words_from_preds):

        for i in words_from_preds:

            if (self.dictionary_words.has_key(i)):
                self.dictionary_words[i] += 1
            else:
                self.dictionary_words[i] = 1

        return True

    def update_dict_from_data(self):

        while (self.load_data()):
            self.update_dict_from_video()

        self.save_h5data('dictionary_keys', self.dictionary_words.keys())
        self.save_h5data('dictionary_values', self.dictionary_words.values())

        return True

    def update_dict_from_video(self):

        next_set_from_cubarray,stats = self.fetch_next_set_from_cubarray()

        while (stats):
            surroundings_from_three_rows = self.create_surroundings(three_rows_cubarray=next_set_from_cubarray)
            predictions = self.predict_on_surroundings(surroundings_from_three_rows)
            words_from_preds = self.create_words_from_predictions(predictions)
            self.update_dict_with_words(words_from_preds)

            next_set_from_cubarray,stats = self.fetch_next_set_from_cubarray()

        return True

    def print_details_and_plot(self, graph_name_frq):

        print "########################################"
        print "MAXIMUM FREQUENCY OF WORDS:"
        print max(self.dictionary_words.values())
        print "########################################"

        print "########################################"
        print "MINIMUM FREQUENCY OF WORDS:"
        print min(self.dictionary_words.values())
        print "########################################"


        hfm, = plt.plot(self.dictionary_words.values(), label='frequency_words')
        plt.legend(handles=[hfm])
        plt.title('Word Frequency')
        plt.xlabel('Word_index')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.model_store, graph_name_frq), bbox_inches='tight')
        plt.close()


        return True

    def __del__(self):
        print ("Destructor called for TrainDictionary")

class TestDictionary:

    def __init__(self,model,data_store,data_test_h5=None,notest=False,model_store=None,test_loss_metric='dssim',use_dist_in_word=False, nc=3, round_to=1,use_basis_dict=False):

        self.model = None
        self.dictionary_words = {}
        self.means = None

        self.data_store = None          # Where the test data is stored

        self.vid = None                 # Id of the video being processed ATM -> Always points to the next id to be loaded
        self.cubarray = None            # Array of all cuboids sampled in a video, arranged in a spatio-temporal fashion
        self.amongtarray = None         # Contains anomaly groundtruths of corresponding cuboids in cubarray (shape = cubarray.shape())
        self.anompercentarray = None    # Contains the anomaly percentage in the cuboid
        self.pixmaparray = None         # Each element contains a tuple of the pixel centres of corresponding cuboid in cubarray
                                        # (shape = cubarray.shape())

        self.cubarray_process_index = None  # The index of the set in cubarray that is being processed.

        self.list_of_cub_words_full_dataset = []
        self.list_of_cub_loss_full_dataset = []
        self.list_of_cub_anom_gt_full_dataset = []
        self.list_of_cub_anomperc_full_dataset = []

        self.list_full_dset_cuboid_frequencies = []
        self.list_anom_gifs = []
        self.list_full_dset_dist = []

        self.notest = notest

        self.timesteps = 8
        self.test_loss_metric = test_loss_metric
        self.data_store = data_store

        self.data_vc = data_test_h5[0]
        self.data_va = data_test_h5[1]
        self.data_vp = data_test_h5[2]
        self.data_ap = data_test_h5[3]

        self.use_dist_in_word=use_dist_in_word
        self.round_to=round_to

        self.n_channels = nc
        self.model_store = model_store
        self.image_store = os.path.join(model_store,'test_images')

        if(not os.path.exists(self.image_store)):
            os.mkdir(self.image_store)

        self.use_basis_dict = use_basis_dict

        if(self.use_basis_dict):
            self.basis_dict = pickle.load(open(os.path.join(self.model_store,'basis_dict.pkl'), 'rb'))
            self.basis_dict_comp = self.basis_dict.components_
            self.list_of_dict_recon_full_dataset=[]

        if(not notest):
            self.model_enc = model.encoder
            self.model_ae = model.ae

            self.means = model.means  #means of each cluster learned by the model

            self.vid = 1

        if(notest):


            self.list_of_cub_anom_gt_full_dataset = self.load_h5data('list_cub_anomgt_full_dataset')

            self.list_of_cub_loss_full_dataset = self.load_h5data('list_cub_loss_full_dataset')

            self.list_full_dset_dist = self.load_h5data('list_dist_measure_full_dataset')

            self.list_of_cub_anomperc_full_dataset = self.load_h5data('list_cub_anompercentage_full_dataset')

            if(self.use_basis_dict):
                self.list_of_dict_recon_full_dataset = self.load_h5data('list_of_dict_recon_full_dataset')

    def load_data(self):

        if('video_cuboids_array_'+str(self.vid) in self.data_vc.keys()):
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$"
            print "LOADING VIDEO ARRAY ", self.vid
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            self.cubarray = np.array(self.data_vc.get('video_cuboids_array_'+str(self.vid)))
            print self.cubarray.shape
            self.anomgtarray = np.array(self.data_va.get('video_anomgt_array_' + str(self.vid)))
            print self.anomgtarray.shape
            self.pixmaparray = np.array(self.data_vp.get('video_pixmap_array_' + str(self.vid)))
            print self.pixmaparray.shape
            self.anompercentarray = np.array(self.data_ap.get('video_anomperc_array_'+str(self.vid)))
            print self.anompercentarray.shape

            self.vid+=1
            self.cubarray_process_index = 1
            return True

        else:
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$"
            print "VIDEOS FINISHED"
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            return False

    def save_h5data(self,name,content):

        with h5py.File(os.path.join(self.model_store, name+'.h5'), "a") as f:

            if (name in f.keys()):
                del f[name]

            dset = f.create_dataset(name, data=content)

        return True

    def load_h5data(self,name):
        dset = h5py.File(os.path.join(self.model_store, name+'.h5'), 'r')
        content = list(dset.get(name))
        dset.close()
        return content

    def fetch_next_set_from_cubarray(self):
        #returns 3 things. 3 rows of cubarray, The relevant row of anomaly_gt and the relevant row of  pixmap
        if(self.cubarray_process_index<len(self.cubarray)-1):
            print self.cubarray_process_index,'/',len(self.cubarray)-1
            sys.stdout.write("\033[F")
            three_rows_cubarray = self.cubarray[self.cubarray_process_index-1:self.cubarray_process_index+2]
            relevant_row_anom_gt = self.anomgtarray[self.cubarray_process_index]
            relevant_row_anompercentage = self.anompercentarray[self.cubarray_process_index]
            relevant_row_pixmap = self.pixmaparray[self.cubarray_process_index]

            self.cubarray_process_index+=1

            return (three_rows_cubarray,relevant_row_anom_gt,relevant_row_pixmap,relevant_row_anompercentage)
        else:
            return False

    def create_surroundings(self,three_rows_cubarray,relevant_row_anom_gt,relevant_row_anompercentage,gif=False):

        # start = time.time()
        rows = three_rows_cubarray[0].shape[0]
        cols = three_rows_cubarray[0].shape[1]

        list_surrounding_cuboids_from_three_rows = []
        sublist_full_dataset_anom_gt = []
        sublist_full_dataset_anompercentage = []
        sublist_full_dataset_dssim_loss = []
        sublist_full_dataset_distances = []


        for j in range(1, rows - 1):
            for k in range(1, cols - 1):
                surroundings = []

                surr_idx = 0

                current_cuboid = three_rows_cubarray[1][j, k]

                sublist_full_dataset_anom_gt.append(relevant_row_anom_gt[j,k])
                sublist_full_dataset_anompercentage.append(relevant_row_anompercentage[j,k])

                if(not gif):
                    if(self.test_loss_metric=='dssim'):
                        loss = self.model_ae.evaluate(x=np.expand_dims(current_cuboid,0),y=None,verbose=False)
                    else:
                        loss = mean_squared_error(np.expand_dims(current_cuboid,0),
                                                  self.model_ae.predict(np.expand_dims(current_cuboid,0)))

                    sublist_full_dataset_dssim_loss.append(loss)

                surroundings.append(current_cuboid)

                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j, k])
                surroundings.append(three_rows_cubarray[surr_idx][j, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k + 1])

                surr_idx = 1
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k + 1])

                surr_idx = 2
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j - 1, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j, k])
                surroundings.append(three_rows_cubarray[surr_idx][j, k + 1])

                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k - 1])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k])
                surroundings.append(three_rows_cubarray[surr_idx][j + 1, k + 1])

                list_surrounding_cuboids_from_three_rows.append(np.array(surroundings))

                if(not gif):
                    encoded = self.model_enc.predict(np.array(surroundings))
                    d = np.min(cdist(encoded,self.means),axis=1)
                    sublist_full_dataset_distances.append(d)

                if(self.use_basis_dict and not gif):
                    recon = np.dot(self.basis_dict.transform(encoded[0:1]),self.basis_dict_comp)
                    self.list_of_dict_recon_full_dataset.append(np.linalg.norm(x=(encoded[0:1]-recon)))

        # end = time.time()
        # print (end-start)/60.0, " minutes"
        return np.array(list_surrounding_cuboids_from_three_rows), sublist_full_dataset_anom_gt, \
               sublist_full_dataset_dssim_loss, sublist_full_dataset_distances, sublist_full_dataset_anompercentage

    def predict_on_surroundings(self,surroundings_from_three_rows):

        list_predictions = []

        for i in range(0,len(surroundings_from_three_rows)):
            list_predictions.append(self.model_enc.predict(surroundings_from_three_rows[i]))

        return np.array(list_predictions)

    def create_words_from_predictions(self,predictions,sublist_full_dataset_distances):

        list_words = []

        if(self.use_dist_in_word):
            for i in range(0,len(predictions)):
                x = predictions[i]
                dist = cdist(x,self.means)
                word = tuple(np.hstack((np.argmin(dist,1),np.around(sublist_full_dataset_distances[i],self.round_to))))
                list_words.append(word)
        else:
            for i in range(0,len(predictions)):
                x = predictions[i]
                dist = cdist(x,self.means)
                word = tuple(np.argmin(dist,1))
                list_words.append(word)

        return list_words

    def update_dict_with_words(self,words_from_preds):

        for i in words_from_preds:

            if(self.dictionary_words.has_key(i)):
                self.dictionary_words[i]+=1
            else:
                self.dictionary_words[i]=1

        return True

    def process_data(self):

        while (self.load_data()):
            self.process_video()


        self.save_h5data('list_cub_anomgt_full_dataset',self.list_of_cub_anom_gt_full_dataset)

        self.save_h5data('list_cub_loss_full_dataset',self.list_of_cub_loss_full_dataset)

        self.save_h5data('list_dist_measure_full_dataset',self.list_full_dset_dist)

        self.save_h5data('list_cub_anompercentage_full_dataset',self.list_of_cub_anomperc_full_dataset)

        if(self.use_basis_dict):
            self.save_h5data('list_of_dict_recon_full_dataset', self.list_of_dict_recon_full_dataset)

        return True

    def process_video(self):

        next_set_from_cubarray = self.fetch_next_set_from_cubarray()

        while(next_set_from_cubarray):

            surroundings_from_three_rows, sublist_full_dataset_anom_gt, sublist_full_dataset_dssim_loss,\
            sublist_full_dataset_distances, sublist_full_dataset_anompercentage = self.create_surroundings(
                                                                             three_rows_cubarray=next_set_from_cubarray[0],
                                                                             relevant_row_anom_gt=next_set_from_cubarray[1],
                                                                             relevant_row_anompercentage=next_set_from_cubarray[3])

            self.list_of_cub_anom_gt_full_dataset.extend(sublist_full_dataset_anom_gt)
            self.list_of_cub_loss_full_dataset.extend(sublist_full_dataset_dssim_loss)
            self.list_full_dset_dist.extend(sublist_full_dataset_distances)
            # self.list_of_cub_words_full_dataset.extend(words_from_preds)
            self.list_of_cub_anomperc_full_dataset.extend(sublist_full_dataset_anompercentage)

            next_set_from_cubarray = self.fetch_next_set_from_cubarray()

        return True

    def print_details_and_plot(self,graph_name_frq,graph_name_loss):

        print "########################################"
        print "MAXIMUM FREQUENCY OF WORDS:"
        print max(self.dictionary_words.values())
        print "########################################"

        print "########################################"
        print "MINIMUM FREQUENCY OF WORDS:"
        print min(self.dictionary_words.values())
        print "########################################"

        print "########################################"
        print "MAXIMUM DSSIM LOSS:"
        print max(self.list_of_cub_loss_full_dataset)
        print "########################################"

        print "########################################"
        print "MINIMUM DSSIM LOSS:"
        print min(self.list_of_cub_loss_full_dataset)
        print "########################################"

        hfm, = plt.plot(self.dictionary_words.values(), label='frequency_words')
        plt.legend(handles=[hfm])
        plt.title('Word Frequency')
        plt.xlabel('Word_index')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.image_store, graph_name_frq), bbox_inches='tight')
        plt.close()

        hfm, = plt.plot(self.list_of_cub_loss_full_dataset, label='loss values')
        plt.legend(handles=[hfm])
        plt.title('DSSIM Reconstruction Loss')
        plt.xlabel('Cuboid Index')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.image_store, graph_name_loss), bbox_inches='tight')
        plt.close()

        return True

    def make_list_full_dset_cuboid_frequencies(self):

        if (self.notest and os.path.exists(os.path.join(self.model_store, 'list_cub_frequencies_full_dataset.h5'))):
            print "LIST CUBOID FREQUENCIES ALREADY EXISTS."
            self.list_full_dset_cuboid_frequencies = self.load_h5data('list_cub_frequencies_full_dataset')

            return True

        self.list_full_dset_cuboid_frequencies=[]

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "MAKING LIST CUBOID FREQUENCIES"
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

        for i in self.list_of_cub_words_full_dataset:
            if (i in self.dictionary_words.keys()):
                self.list_full_dset_cuboid_frequencies.append(self.dictionary_words[i])
            else:
                self.list_full_dset_cuboid_frequencies.append(0)


        self.save_h5data('list_cub_frequencies_full_dataset', self.list_full_dset_cuboid_frequencies)

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "LEN LIST_CUB_FREQUENCIES_FULL", len(self.list_full_dset_cuboid_frequencies)
        print "MAX LIST_CUB_FREQUENCIES_FULL", max(self.list_full_dset_cuboid_frequencies)
        print "MIN LIST_CUB_FREQUENCIES_FULL", min(self.list_full_dset_cuboid_frequencies)
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

        return True

    def make_p_r_f_a_curve(self,array_to_th,lt=False,prfa_graph_name='prf.png',tp_fp_graph_name='tpfp.png',deets_filename='prf_deets',metric='metric'):

        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        accuracy_score_list = []

        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []

        lspace = np.linspace(min(array_to_th), max(array_to_th), 3000)
        total_num_tp = np.sum(self.list_of_cub_anom_gt_full_dataset)

        print "#################################################################################"
        print "TOTAL NUMBER OF TRUE POSITIVES:" , total_num_tp
        print "TOTAL NUMBER OF SAMPLES TO BE TESTED:",len(self.list_of_cub_anom_gt_full_dataset)
        print "RATIO:",total_num_tp/(len(self.list_of_cub_anom_gt_full_dataset)+0.0)
        print "ACC WHEN ALL 0: ", (len(self.list_of_cub_anom_gt_full_dataset) - total_num_tp)/(len(self.list_of_cub_anom_gt_full_dataset)+0.0) * 100
        print "#################################################################################"


        for i in tqdm(lspace):

            y_true = np.array(self.list_of_cub_anom_gt_full_dataset)

            if(lt):
                y_pred = (np.array(array_to_th)<=i)
            else:
                y_pred = (np.array(array_to_th)>=i)

            cm = confusion_matrix(y_true,y_pred)

            precision_score_list.append(precision_score(y_true,y_pred)*100)
            recall_score_list.append(recall_score(y_true,y_pred)*100)
            f1_score_list.append(f1_score(y_true,y_pred)*100)
            accuracy_score_list.append(accuracy_score(y_true,y_pred)*100)

            TN = cm[0][0]

            FP = cm[0][1]

            FN = cm[1][0]

            TP = cm[1][1]

            tp_list.append(TP)
            tn_list.append(TN)

            fn_list.append(FN)
            fp_list.append(FP)

        print "##########################################################################"
        print "MAX ACCURACY:",max(accuracy_score_list)
        print "THRESHOLD:",lspace[accuracy_score_list.index(max(accuracy_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX PRECISION:",max(precision_score_list)
        print "THRESHOLD:",lspace[precision_score_list.index(max(precision_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX RECALL:", max(recall_score_list)
        print "THRESHOLD:", lspace[recall_score_list.index(max(recall_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX F1:", max(f1_score_list)
        print "THRESHOLD:", lspace[f1_score_list.index(max(f1_score_list))]
        print "##########################################################################"

        #Plot PRFA curve
        hfm, = plt.plot(precision_score_list, label='precision_score')
        hfm2, = plt.plot(recall_score_list, label='recall_score')
        hfm3, = plt.plot(f1_score_list, label='f1_score')
        hfm4, = plt.plot(accuracy_score_list,label='accuracy_score')

        s_acc = 'max_acc:'+str(format(max(accuracy_score_list),'.2f'))+' at '+ str(float(lspace[accuracy_score_list.index(max(accuracy_score_list))]))
        s_f1 = 'max_f1:'+str(format(max(f1_score_list),'.2f'))+' at '+ str(float(lspace[f1_score_list.index(max(f1_score_list))]))
        s_pr = 'max_pr:'+str(format(max(precision_score_list),'.2f'))+' at '+ str(float(lspace[precision_score_list.index(max(precision_score_list))]))
        s_re = 'max_re:' + str(format(max(recall_score_list),'.2f')) + ' at ' + str(float(lspace[recall_score_list.index(max(recall_score_list))]))


        plt.legend(handles=[hfm,hfm2,hfm3,hfm4])
        plt.title('Precision, Recall, F1_Score')
        plt.ylabel('Scores')
        plt.xlabel(metric)
        plt.savefig(os.path.join(self.image_store, prfa_graph_name), bbox_inches='tight')
        plt.close()

        f = open(os.path.join(self.image_store, deets_filename), 'a+')
        f.write(s_acc +  '\n')
        f.write(s_f1  +  '\n')
        f.write(s_pr  +  '\n')
        f.write(s_re  +  '\n')
        f.close()

        #Plot TPFPcurve
        hfm,  = plt.plot(tp_list,  label='N_true_positives')
        hfm2, = plt.plot(fp_list, label='N_false_positives')
        hfm3, = plt.plot(tn_list, label='N_true_negatives')
        hfm4, = plt.plot(fn_list, label='N_false_negatives')

        plt.legend(handles=[hfm,hfm2,hfm3,hfm4])
        plt.title('TP,FP,TN,FN')
        plt.ylabel('Values')
        plt.xlabel(metric)
        plt.savefig(os.path.join(self.image_store, tp_fp_graph_name), bbox_inches='tight')
        plt.close()

        score_dict = {'max_acc': float(max(accuracy_score_list)), 'max_f1': float(max(f1_score_list)),
                      'max_pre': float(max(precision_score_list)), 'max_rec': float(max(recall_score_list))}

        return score_dict

    def evaluate_prfa_dict_recon(self,comps=10):

        score_dict = self.evaluate_prfa(self.list_of_dict_recon_full_dataset,lt=False)
        self.plot_basis_dict_recon_measure_of_samples('basis_dict_recon_plot_samples_'+str(comps)+'.png')

        return score_dict

    def evaluate_prfa(self,array_to_th,lt=False):

        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        accuracy_score_list = []

        lspace = np.linspace(min(array_to_th), max(array_to_th), 2500)
        total_num_tp = np.sum(self.list_of_cub_anom_gt_full_dataset)

        print "#################################################################################"
        print "TOTAL NUMBER OF TRUE POSITIVES:" , total_num_tp
        print "TOTAL NUMBER OF SAMPLES TO BE TESTED:",len(self.list_of_cub_anom_gt_full_dataset)
        print "RATIO:",total_num_tp/(len(self.list_of_cub_anom_gt_full_dataset)+0.0)
        print "ACC WHEN ALL 0: ", (len(self.list_of_cub_anom_gt_full_dataset) - total_num_tp)/(len(self.list_of_cub_anom_gt_full_dataset)+0.0) * 100
        print "#################################################################################"


        for i in tqdm(lspace):

            y_true = np.array(self.list_of_cub_anom_gt_full_dataset)

            if(lt):
                y_pred = (np.array(array_to_th)<=i)
            else:
                y_pred = (np.array(array_to_th)>=i)


            precision_score_list.append(precision_score(y_true,y_pred)*100)
            recall_score_list.append(recall_score(y_true,y_pred)*100)
            f1_score_list.append(f1_score(y_true,y_pred)*100)
            accuracy_score_list.append(accuracy_score(y_true,y_pred)*100)


        print "##########################################################################"
        print "MAX ACCURACY:",max(accuracy_score_list)
        print "THRESHOLD:",lspace[accuracy_score_list.index(max(accuracy_score_list))]
        max_acc_threshold = lspace[accuracy_score_list.index(max(accuracy_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX PRECISION:",max(precision_score_list)
        print "THRESHOLD:",lspace[precision_score_list.index(max(precision_score_list))]
        max_pre_threshold = lspace[precision_score_list.index(max(precision_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX RECALL:", max(recall_score_list)
        print "THRESHOLD:", lspace[recall_score_list.index(max(recall_score_list))]
        max_re_threshold = lspace[recall_score_list.index(max(recall_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX F1:", max(f1_score_list)
        print "THRESHOLD:", lspace[f1_score_list.index(max(f1_score_list))]
        max_f1_threshold = lspace[f1_score_list.index(max(f1_score_list))]
        print "##########################################################################"

        score_dict = {'max_acc': float(max(accuracy_score_list)), 'max_f1': float(max(f1_score_list)),
                      'max_pre': float(max(precision_score_list)), 'max_rec': float(max(recall_score_list)),
                      'max_acc_th': max_acc_threshold, 'max_f1_th': max_f1_threshold, 'max_pre_th':max_pre_threshold,
                      'max_rec_th': max_re_threshold}

        return score_dict

    def evaluate_prfa_on_specific(self,array_to_th,gt_array,lt=False):

        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        accuracy_score_list = []
        cm_list = []

        lspace = np.linspace(min(array_to_th), max(array_to_th), 4000)
        total_num_tp = np.sum(gt_array)

        print "#################################################################################"
        print "TOTAL NUMBER OF TRUE POSITIVES:" , total_num_tp
        print "TOTAL NUMBER OF SAMPLES TO BE TESTED:",len(gt_array)
        print "RATIO:",total_num_tp/(len(gt_array)+0.0)
        print "ACC WHEN ALL 0: ", (len(gt_array) - total_num_tp)/(len(gt_array)+0.0) * 100
        print "#################################################################################"


        for i in tqdm(lspace):

            y_true = gt_array

            if(lt):
                y_pred = (array_to_th<=i)
            else:
                y_pred = (array_to_th>=i)


            precision_score_list.append(precision_score(y_true,y_pred)*100)
            recall_score_list.append(recall_score(y_true,y_pred)*100)
            f1_score_list.append(f1_score(y_true,y_pred)*100)
            cm_list.append(confusion_matrix(y_true,y_pred))
            accuracy_score_list.append(accuracy_score(y_true,y_pred)*100)


        print "##########################################################################"
        print "MAX ACCURACY:",max(accuracy_score_list)
        print "THRESHOLD:",lspace[accuracy_score_list.index(max(accuracy_score_list))]
        max_acc_threshold = lspace[accuracy_score_list.index(max(accuracy_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX PRECISION:",max(precision_score_list)
        print "THRESHOLD:",lspace[precision_score_list.index(max(precision_score_list))]
        max_pre_threshold = lspace[precision_score_list.index(max(precision_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX RECALL:", max(recall_score_list)
        print "THRESHOLD:", lspace[recall_score_list.index(max(recall_score_list))]
        max_re_threshold = lspace[recall_score_list.index(max(recall_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX F1:", max(f1_score_list)
        print "THRESHOLD:", lspace[f1_score_list.index(max(f1_score_list))]
        max_f1_threshold = lspace[f1_score_list.index(max(f1_score_list))]
        print "##########################################################################"
        print "CONFUSION_MATRIX FOR MAX_F1"
        print cm_list[f1_score_list.index(max(f1_score_list))]
        print "##########################################################################"

        score_dict = {'max_acc': float(max(accuracy_score_list)), 'max_f1': float(max(f1_score_list)),
                      'max_pre': float(max(precision_score_list)), 'max_rec': float(max(recall_score_list)),
                      'max_acc_th': max_acc_threshold, 'max_f1_th': max_f1_threshold, 'max_pre_th':max_pre_threshold,
                      'max_rec_th': max_re_threshold}

        return score_dict

    def plot_roc_curve(self,plot_title,array_scores,array_gt,plot_filename):

        message_print("START PLOTTING ROC CURVE")
        message_print(plot_title)
        fpr,tpr ,_ = roc_curve(y_true=array_gt,y_score=array_scores)
        auc_score = auc(fpr,tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(plot_title)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.image_store,plot_filename),bbox_inches='tight')
        plt.close()

        return True

    def make_comparitive_plot(self,graph_name,array_to_consider,metric_name=None,lt=False):

        aclist = array_to_consider.tolist()
        score_dict = self.evaluate_prfa(array_to_th=aclist)

        threshold = score_dict['max_f1_th']

        y_true = np.array(self.list_of_cub_anom_gt_full_dataset)

        y_perc = np.array(self.list_of_cub_anomperc_full_dataset)

        args_arr_sort = np.argsort(array_to_consider)

        y_true_arr = y_true[args_arr_sort]
        y_perc_arr = y_perc[args_arr_sort]

        array_to_consider = array_to_consider[args_arr_sort]

        pdf_name = graph_name.split('.')[0] + '.pdf'

        with PdfPages(pdf_name) as pdf:

            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(40, 40), sharex='col', sharey='row')
            plt.suptitle('Distribution of ' +metric_name+' scores from the dataset. Green:Scores of Normal Cuboids, Red:Scores of Anomaly Cuboids', fontsize=30)
            plt.setp(ax.get_xticklabels(), visible=True)
            sns.distplot(array_to_consider[y_true_arr==0], kde=False, rug=False, hist=True, ax=ax,color='green')
            sns.distplot(array_to_consider[y_true_arr==1], kde=False, rug=False, hist=True, ax=ax,color='red')
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

        colors = ['green', 'red']

        f, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(80, 80))

        ax1 = ax[0,0]
        ax2 = ax[0,1]
        ax3 = ax[1,1]

        im1 = ax1.scatter(range(0,len(array_to_consider)),array_to_consider,c=y_true_arr, cmap=ListedColormap(colors),alpha=0.5)
        ax1.set_title('ANOMS:Red, N-ANOMS:Green')
        ax1.axhline(y=threshold, label='best threshold', color='b')
        ax1.set_ylabel(metric_name)
        ax1.set_xlabel('Cuboid index')
        ax1.grid(True)
        cb1 = f.colorbar(im1,ax=ax1)
        loc = np.arange(0, max(y_true), max(y_true) / float(len(colors)))
        cb1.set_ticks(loc)
        cb1.set_ticklabels(['normal','anomaly'])

        arr_anoms = array_to_consider[y_true_arr==1]

        im2 = ax2.scatter(range(0,len(arr_anoms)),arr_anoms,c='red',alpha=0.5)
        ax2.axhline(y=threshold, label='best threshold', color='b')
        ax2.set_title('ANOMS:Red')
        ax2.set_ylabel(metric_name)
        ax2.set_xlabel('Cuboid index')
        ax2.grid(True)

        arr_perc = y_perc_arr[y_true_arr==1]

        im3 = ax3.scatter(range(0, len(arr_perc)), arr_perc, c='blue', alpha=0.5)
        ax3.set_title('ANOMPERC')
        ax3.set_ylabel('percent of anomaly pixels in cuboid')
        ax3.set_xlabel('Cuboid index')
        ax3.grid(True)


        plt.savefig(os.path.join(self.image_store, graph_name), bbox_inches='tight')
        plt.close()

        self.write_prf_details_to_file(filename='prf_details.txt',score_dict=score_dict,metric_name=metric_name)
        self.plot_roc_curve(plot_title='Reciever Operating Characteristic when using '+metric_name.upper(),
                            array_scores=(array_to_consider),
                            array_gt=y_true_arr,
                            plot_filename='roc_for_'+metric_name+'.png'
                            )
        return True

    def plot_frequencies_of_samples(self,anom_frequency_graph_name):

        frequency_array = np.array(self.list_full_dset_cuboid_frequencies)
        self.make_comparitive_plot(anom_frequency_graph_name,frequency_array,'Frequency',lt=True)

        return True

    def plot_loss_of_samples(self,recon_loss_graph_name):

        loss_array = np.array(self.list_of_cub_loss_full_dataset)
        self.make_comparitive_plot(recon_loss_graph_name,loss_array,self.test_loss_metric+'-loss',lt=False)

        return True

    def plot_distance_measure_of_samples(self,distance_measure_samples_graph_name,dmeasure='mean'):

        if(dmeasure=='mean'):
            dmeasure_array = np.mean(np.array(self.list_full_dset_dist),axis=1)

        elif(dmeasure=='meanxloss'):
            dmeasure_array = np.mean(np.array(self.list_full_dset_dist),axis=1) * np.array(self.list_of_cub_loss_full_dataset)

        elif(dmeasure=='distance'):
            dmeasure_array = np.array(self.list_full_dset_dist)[:,0]

        elif(dmeasure=='distancexloss'):

            dmeasure_array = np.array(self.list_full_dset_dist)[:, 0] * np.array(self.list_of_cub_loss_full_dataset)
        else:
            print "ERROR: DMEASURE MUST BE = one of [mean, meanxloss, distance, distancexloss]"
            return False

        self.make_comparitive_plot(distance_measure_samples_graph_name,dmeasure_array,metric_name=dmeasure,lt=False)

        return True

    def plot_basis_dict_recon_measure_of_samples(self,basis_dict_recon_measure_samples_graph_name):

        if(self.use_basis_dict):
            self.make_comparitive_plot(basis_dict_recon_measure_samples_graph_name, np.array(self.list_of_dict_recon_full_dataset),
                                       metric_name='Basis_Dict Reconstruction Error',lt=False)
        else:
            print "use_basis_dict is False"

        return True

    def create_pdf_distance_surroundings(self,list_distances,pdf_name):

        plots_done = 0

        l = list_distances.shape[1]


        with PdfPages(pdf_name) as pdf:

            while (plots_done < len(list_distances)):

                    f, ax = plt.subplots(10, 2, sharex='col', sharey='row', figsize=(40, 30))

                    for i in range(0,10):
                        for j in range(0,2):

                            ax[i][j].plot(list_distances[plots_done])
                            plots_done+=1

                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

        return True

    def create_distance_metric_pdfs(self,num_plots_per_pdf):

        #distance_normal_cuboids_pdf
        list_normal_cuboid_distances = np.array(self.list_full_dset_dist)[np.array(self.list_of_cub_anom_gt_full_dataset)==0]
        list_normal_cuboid_distances = list_normal_cuboid_distances[np.random.randint(0,len(list_normal_cuboid_distances),num_plots_per_pdf)]
        self.create_pdf_distance_surroundings(list_distances=list_normal_cuboid_distances,pdf_name=os.path.join(self.image_store,'cubds_norm_dist_bar.pdf'))

        # distance_anom_cuboids_pdf
        list_anom_cuboid_distances = np.array(self.list_full_dset_dist)[np.array(self.list_of_cub_anom_gt_full_dataset)==1]
        list_anom_cuboid_distances = list_anom_cuboid_distances[np.random.randint(0,len(list_anom_cuboid_distances),num_plots_per_pdf)]
        self.create_pdf_distance_surroundings(list_distances=list_anom_cuboid_distances,pdf_name=os.path.join(self.image_store,'cubds_anom_dist_bar.pdf'))

        return True

    def make_anomaly_gifs(self):

        if not os.path.exists(os.path.join(self.model_store,'anom_gifs')):
            os.makedirs(os.path.join(self.model_store,'anom_gifs'))

        self.vid=1 #Reset vid

        while (self.load_data()):
            self.create_anom_gif_from_video()

        return True

    def create_anom_gif_from_video(self):

        next_set_from_cubarray = self.fetch_next_set_from_cubarray()

        while(next_set_from_cubarray):
            surroundings_from_three_rows, sublist_full_dataset_anom_gt, sublist_full_dataset_dssim_loss, sublist_full_dataset_distances, \
            sublist_full_dataset_anompercentage = self.create_surroundings(three_rows_cubarray=next_set_from_cubarray[0],
                                                                           relevant_row_anom_gt=next_set_from_cubarray[1],
                                                                           relevant_row_anompercentage=next_set_from_cubarray[3],
                                                                           gif=True)

            if(not any(sublist_full_dataset_anom_gt)):
                next_set_from_cubarray = self.fetch_next_set_from_cubarray()
                continue
            else:
                if(np.random.rand()<0.1):
                    surroundings_from_three_rows = surroundings_from_three_rows[np.array(sublist_full_dataset_anom_gt)==True]
                    predictions = self.predict_on_surroundings(surroundings_from_three_rows)
                    words_from_preds = self.create_words_from_predictions(predictions,sublist_full_dataset_distances)

                    self.make_gifs_from_surroundings(surroundings_from_three_rows,words_from_preds)


            next_set_from_cubarray = self.fetch_next_set_from_cubarray()


        return True

    def make_gifs_from_surroundings(self,surroundings_from_three_rows,words_from_preds):

        for idx,i in enumerate(surroundings_from_three_rows):

            past    = i[1:10]
            present = i[[10,11,12,13,0,14,15,16,17],:]
            future  = i[18:27]

            word_of_cuboid = words_from_preds[idx]

            frequency_of_cuboid = 0
            if (self.dictionary_words.has_key(word_of_cuboid)):
                frequency_of_cuboid = self.dictionary_words[word_of_cuboid]
            else:
                print "KEY ERROR.", "KEY:",word_of_cuboid, "DOES NOT EXIST IN THE DICTIONARY."

            self.save_images_for_gif(past,'PAST')
            self.save_images_for_gif(present,'PRESENT')
            self.save_images_for_gif(future,'FUTURE')

            self.create_save_gif(frequency_of_cuboid)

        return True

    def save_images_for_gif(self,cuboids,time):

        multiplier = 0
        timesteps = cuboids[0].shape[-1]/self.n_channels
        assert (time=='PAST' or time=='PRESENT' or time=='FUTURE')

        if(time=='PAST'):
            multiplier = 0
        elif(time=='PRESENT'):
            multiplier=1
        elif(time=='FUTURE'):
            multiplier=2

        for i in range(0,timesteps):
            f, ax_array = plt.subplots(3, 3)

            for idx in range(0,9):

                row = int(idx/3)
                col = idx%3
                ax_array[row,col].imshow(np.uint8(cuboids[idx,:,:,i*self.n_channels:(i+1)*self.n_channels]*255.0))
                ax_array[row,col].set_axis_off()
                if(idx==4):
                    ax_array[row, col].set_title(time)

            plt.axis('off')
            plt.savefig(os.path.join(self.model_store,'anom_gifs', str(timesteps*multiplier+i)+'.png'), bbox_inches='tight')
            plt.close()

    def create_save_gif(self,freq):

        name = os.path.join(self.model_store, 'anom_gifs', str(freq) + '.gif')

        images = []
        for i in range(0,self.timesteps*3):
            images.append(imageio.imread(os.path.join(self.model_store,'anom_gifs', str(i) + '.png')))

        if(os.path.exists(name)):
            j = 1

            while (os.path.exists(name)):
                j+=1
                name = os.path.join(self.model_store, 'anom_gifs', str(freq) + '_' + str(j) + '.gif')

        imageio.mimsave(name, images)

        return True

    def feature_analysis_normvsanom(self):

        self.vid=1 #Reset vid

        full_list_feats_normal = []
        full_list_feats_anomaly = []

        while (self.load_data()):

            feats_normal, feats_anomaly = self.create_feats_to_analyze_from_video()
            full_list_feats_normal.extend(feats_normal)
            full_list_feats_anomaly.extend(feats_anomaly)


        full_list_feats_normal = np.array(full_list_feats_normal)
        full_list_feats_anomaly = np.array(full_list_feats_anomaly)


        pdf_name = os.path.join(self.model_store, 'features_kde_normvsanom.pdf')

        if (os.path.exists(pdf_name)):
            os.remove(pdf_name)
        cols_plots = 16
        rows_plots = (full_list_feats_normal.shape[1] / cols_plots)

        with PdfPages(pdf_name) as pdf:

            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            fig, ax = plt.subplots(ncols=cols_plots, nrows=rows_plots, figsize=(200, 200), sharex='col', sharey='row')
            plt.suptitle('Distribution of each feature in the dataset. Green:Features of Normal Cuboids, Red: Features of Anomaly Cuboids', fontsize=30)

            for i in range(0, full_list_feats_normal.shape[1]):
                print "PROCESSING FEATURE:", i
                try:
                    ax[int(i / cols_plots)][i % cols_plots].set_title('feature: ' + str(i + 1), fontsize=20)
                    plt.setp(ax[int(i / cols_plots)][i % cols_plots].get_xticklabels(), visible=True)

                    sns.distplot(full_list_feats_normal[:, i], kde=False, rug=False, hist=True,
                                 ax=ax[int(i / cols_plots)][i % cols_plots],color='green')

                    sns.distplot(full_list_feats_anomaly[:, i], kde=False, rug=False, hist=True,
                             ax=ax[int(i / cols_plots)][i % cols_plots],color='red')
                except:
                    print "SKIPPING FEATURE:", i

            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

        return True

    def return_gm_score(self,feats):

       return self.gm.score_samples(np.array(feats)).tolist()

    def return_maha_dist(self,feats):

        distances = []

        predicted_classes = self.gm.predict(feats)
        mu = self.mu_array[predicted_classes]
        covariance_invs = self.cov_inv[predicted_classes]

        for idx,i in enumerate(feats):
            ph1 = np.matmul((i - mu[idx]), covariance_invs[idx])
            distances.append(np.matmul(ph1, (i - mu[idx]).T))

        return distances

    def gmm_analysis(self):

        self.vid=1 #Reset vid
        self.gm = so.load_obj(os.path.join(self.model_store,'gmm.pkl'))

        self.mu_array = self.gm.means_
        self.cov_array = self.gm.covariances_
        self.cov_inv = np.zeros((self.cov_array.shape))

        for idx, i in enumerate(self.cov_array):
            self.cov_inv[idx] = np.linalg.inv(i)

        full_list_scores_normal = []
        full_list_scores_anomaly = []

        while (self.load_data()):

            feats_normal, feats_anomaly = self.create_feats_to_analyze_from_video()
            full_list_scores_normal.extend(self.return_maha_dist(feats_normal))
            full_list_scores_anomaly.extend(self.return_maha_dist(feats_anomaly))

        full_list_scores_normal = np.sort(np.array(full_list_scores_normal))
        full_list_scores_anomaly = np.sort(np.array(full_list_scores_anomaly))


        pdf_name = os.path.join(self.image_store, 'gmm_analysis.pdf')

        if (os.path.exists(pdf_name)):
            os.remove(pdf_name)


        with PdfPages(pdf_name) as pdf:

            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(40, 40), sharex='col', sharey='row')
            plt.suptitle('Distribution of log-probability scores from the dataset. Green:Scores of Normal Cuboids, Red:Scores of Anomaly Cuboids', fontsize=30)
            plt.setp(ax.get_xticklabels(), visible=True)
            sns.distplot(full_list_scores_normal, kde=False, rug=False, hist=True, ax=ax,color='green')
            sns.distplot(full_list_scores_anomaly, kde=False, rug=False, hist=True, ax=ax,color='red')
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

        list_all_scores = []
        list_all_scores.extend(full_list_scores_normal.tolist())
        list_all_scores.extend(full_list_scores_anomaly.tolist())
        list_all_scores = np.array(list_all_scores)

        list_all_gt = []
        list_all_gt.extend(np.zeros(len(full_list_scores_normal)).tolist())
        list_all_gt.extend(np.ones(len(full_list_scores_anomaly)).tolist())
        list_all_gt = np.array(list_all_gt)

        args_arr_sort = np.argsort(-list_all_scores)

        list_all_gt = list_all_gt[args_arr_sort]
        list_all_scores = list_all_scores[args_arr_sort]

        score_dict = self.evaluate_prfa_on_specific(array_to_th=list_all_scores, gt_array=list_all_gt, lt=False)
        threshold = score_dict['max_f1_th']

        colors = ['green', 'red']

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(80, 40), sharey='row')

        ax1 = ax[0]
        ax2 = ax[1]

        plt.suptitle('Log probability Score plots of anomaly vs normal cuboids', fontsize=30)

        im1 = ax1.scatter(range(0, len(list_all_scores)), list_all_scores, c=list_all_gt,
                          cmap=ListedColormap(colors), alpha=0.5)
        ax1.axhline(y=threshold,label='best threshold',color='b')
        ax1.set_title('ANOMS:Red, N-ANOMS:Green')
        ax1.set_ylabel('Log-probability score')
        ax1.set_xlabel('Cuboid index')
        ax1.grid(True)
        cb1 = fig.colorbar(im1, ax=ax1)
        loc = np.arange(0, max(list_all_gt), max(list_all_gt) / float(len(colors)))
        cb1.set_ticks(loc)
        cb1.set_ticklabels(['normal', 'anomaly'])
        arr_anoms = list_all_scores[list_all_gt == 1]

        im2 = ax2.scatter(range(0, len(arr_anoms)), arr_anoms, c='red', alpha=0.5)
        ax2.axhline(y=threshold, label='best threshold', color='b')
        ax2.set_title('ANOMS:Red')
        ax2.set_ylabel('Log-probability score')
        ax2.set_xlabel('Cuboid index')
        ax2.grid(True)
        plt.savefig(os.path.join(self.image_store, 'gmm_analysis_plot.png'))
        plt.close()

        self.write_prf_details_to_file(filename='prf_details.txt', score_dict=score_dict, metric_name='GMM probability score')

        self.plot_roc_curve(plot_title='Reciever Operating Characteristic when using Gaussian Mixture Model',
                            array_scores=(list_all_scores),
                            array_gt=list_all_gt,
                            plot_filename = 'roc_for_gmm.png')
        return score_dict

    def create_feats_to_analyze_from_video(self):

        list_feats_normal = []
        list_feats_anomaly = []

        next_set_from_cubarray = self.fetch_next_set_from_cubarray()

        while(next_set_from_cubarray):
            surroundings_from_three_rows, sublist_full_dataset_anom_gt, sublist_full_dataset_dssim_loss, sublist_full_dataset_distances, \
            sublist_full_dataset_anompercentage = self.create_surroundings(three_rows_cubarray=next_set_from_cubarray[0],
                                                                           relevant_row_anom_gt=next_set_from_cubarray[1],
                                                                           relevant_row_anompercentage=next_set_from_cubarray[3],
                                                                           gif=True)

            valid_cuboids = surroundings_from_three_rows[:,0]
            predictions = self.model_enc.predict(valid_cuboids)

            feats_normal = list(predictions[np.array(sublist_full_dataset_anom_gt)==False])
            feats_anomaly = list(predictions[np.array(sublist_full_dataset_anom_gt)==True])

            list_feats_normal.extend(feats_normal)
            list_feats_anomaly.extend(feats_anomaly)

            next_set_from_cubarray = self.fetch_next_set_from_cubarray()


        return list_feats_normal, list_feats_anomaly

    def write_prf_details_to_file(self,filename='prf_details.txt',score_dict=None,metric_name = None):

        s_acc = 'max_acc:' + str(format(score_dict['max_acc'], '.5f'))
        s_f1 = 'max_f1:' + str(format(score_dict['max_f1'], '.5f'))
        s_pr = 'max_pr:' + str(format(score_dict['max_pre'], '.5f'))
        s_re = 'max_re:' + str(format(score_dict['max_rec'], '.5f'))

        f = open(os.path.join(self.image_store, filename), 'a+')
        f.write('--------------------------------' + '\n')
        f.write(metric_name + '\n')
        f.write('--------------------------------' + '\n')
        f.write(s_acc +  '\n')
        f.write(s_f1  +  '\n')
        f.write(s_pr  +  '\n')
        f.write(s_re  +  '\n')
        f.write('--------------------------------' + '\n')
        f.close()

        return True

    def __del__(self):
        print ("Destructor called for TestDictionary")

class TestVideoStream:

    def __init__(self, PathToVideos,CubSizeY,CubSizeX,CubTimesteps,ModelStore,Encoder=None,GMM=None,LSTM=False,StridesTime=1,StridesSpace=1,GrayScale=False,BkgSub=True):

        self.PathToVideos = PathToVideos
        self.CubSizeY = CubSizeY
        self.CubSizeX = CubSizeX
        self.CubTimesteps = CubTimesteps
        self.LSTM = LSTM
        self.StridesTime = StridesTime
        self.StridesSpace = StridesSpace
        self.ModelStore = ModelStore
        self.GrayScale = GrayScale
        self.Encoder = Encoder
        self.GMM = GMM

        if(self.GrayScale):
            self.NumChannels = 1
        else:
            self.NumChannels = 3

        self.BkgSub = BkgSub

        self.ListTestImgClass_AllVideos = make_list_test_img_class_all_videos(loc_videos=PathToVideos, n_frames=CubTimesteps, tstrides=StridesTime)

        self.FrameSize_Y, self.FrameSize_X = get_frame_size(self.ListTestImgClass_AllVideos[0][0],self.GrayScale)

    def set_GMMThreshold(self,threshold):

        self.ThresholdGMM = threshold

        return True

    def process_data(self):

        if (not os.path.exists(os.path.join(self.ModelStore, 'Video_Results'))):
            os.mkdir(os.path.join(self.ModelStore, 'Video_Results'))

        for idx,list_test_img_class_video in enumerate(self.ListTestImgClass_AllVideos):

            self.process_video(list_test_img=list_test_img_class_video,idx=idx)

        return True

    def process_video(self,list_test_img,idx):

        if (os.path.exists(os.path.join(self.ModelStore, 'Video_Results', str(idx)))):
            shutil.rmtree(os.path.join(self.ModelStore, 'Video_Results', str(idx)))

        path_save_frames = os.path.join(self.ModelStore, 'Video_Results', str(idx))

        os.mkdir(path_save_frames)

        for test_img in tqdm(list_test_img):
            self.process_self_test_img(test_img,path_save_frames)

        return True

    def process_self_test_img(self,test_img,path_save_frames):

        list_files_past,list_files_present,list_files_future = test_img.get_time_base_filenames()

        array_cuboids_past,_ = self.create_cuboids_from_fnames(list_files_past)
        array_cuboids_present,array_relevant_rows_cols_map_cuboids = self.create_cuboids_from_fnames(list_files_present)
        array_cuboids_future,_ = self.create_cuboids_from_fnames(list_files_future)

        list_array_cuboids = [array_cuboids_past,array_cuboids_present,array_cuboids_future]

        anomaly_map = self.process_list_array_cuboids(list_array_cuboids,array_relevant_rows_cols_map_cuboids)

        self.format_and_save_output_images(anomaly_map,list_files_present,path_save_frames)

        return True

    def format_and_save_output_images(self,anomaly_map,list_files_present,path_save_frames):

        collection = imread_collection(list_files_present,as_grey=True)

        for idx,image in enumerate(collection):

            img = img_as_float(np.uint8(image*255.0))
            img_color = np.dstack((img, img, img))
            img_hsv = color.rgb2hsv(img_color)

            color_mask = np.zeros((img.shape[0], img.shape[1], 3))
            color_mask[:, :, 0] = np.uint8(anomaly_map)
            color_mask_hsv = color.rgb2hsv(color_mask)

            img_hsv[..., 0] = color_mask_hsv[..., 0]
            img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.6

            img_masked = color.hsv2rgb(img_hsv)
            plt.imshow(img_masked)

            top, filename = os.path.split(list_files_present[idx])
            filename, fext = os.path.splitext(filename)
            plt.axis('off')
            plt.savefig(os.path.join(path_save_frames, filename+'.png'),bbox_inches='tight')
            plt.close()

        return True

    def process_list_array_cuboids(self,list_array_cuboids,array_relevant_rows_cols_map_cuboids):

        anomaly_map = np.zeros((self.FrameSize_Y,self.FrameSize_X))

        array_cuboids_present = list_array_cuboids[1]
        array_cuboids_present = array_cuboids_present.reshape(-1,*array_cuboids_present.shape[2:])
        array_relevant_rows_cols_map_cuboids = array_relevant_rows_cols_map_cuboids.reshape(-1,*array_relevant_rows_cols_map_cuboids.shape[2:])

        encodings_of_cuboids = self.Encoder.predict(array_cuboids_present)
        score_samples = self.GMM.score_samples(encodings_of_cuboids)

        thresholded = (score_samples<self.ThresholdGMM)

        for idx,rows_cols_map_cuboid in enumerate(array_relevant_rows_cols_map_cuboids):

            if(thresholded[idx]):
                start_rows = rows_cols_map_cuboid[0]
                end_rows = rows_cols_map_cuboid[1]
                start_cols = rows_cols_map_cuboid[2]
                end_cols = rows_cols_map_cuboid[3]

                anomaly_map[start_rows:end_rows,start_cols:end_cols]=1

        return anomaly_map

    def preprocess_collection(self, image_collection):

        if (self.BkgSub):
            # Subtract mean
            image_collection = image_collection - np.mean(image_collection, axis=0)

            # Rescale data to 0-1 if gs and 0-255 if not-gs
            image_collection = (image_collection - np.min(image_collection))
            image_collection = image_collection / np.max(image_collection)

            if (not self.GrayScale):
                image_collection = np.uint8(image_collection * 255.0)

        if (not self.GrayScale):
            image_collection = image_collection / 255.0

        return image_collection

    def create_cuboids_from_fnames(self,list_files):

        image_collection = imread_collection(list_files, as_grey=self.GrayScale)

        frame_size_y = self.FrameSize_Y
        frame_size_x = self.FrameSize_X

        image_collection = self.preprocess_collection(image_collection)

        lc = np.zeros((frame_size_y, frame_size_x, self.NumChannels * self.CubTimesteps))

        for l in range(0, len(image_collection)):
            lc[:, :, l * self.NumChannels:(l + 1) * self.NumChannels] = image_collection[l].reshape(frame_size_y, frame_size_x, self.NumChannels)

        del image_collection
        image_collection = lc

        list_cuboids_local = []
        list_cuboids_pixmap_local = []

        for j in range(0, frame_size_y,self.StridesSpace):
            for k in range(0, frame_size_x,self.StridesSpace):

                start_rows = int(j - (self.CubSizeY/2))
                end_rows   = int(j + (self.CubSizeY/2))

                start_cols = int(k - (self.CubSizeX/2))
                end_cols   = int(k + (self.CubSizeX/2))


                if (start_rows < 0 or end_rows > frame_size_y or start_cols < 0 or end_cols > frame_size_x):
                    continue

                cuboid_data = image_collection[start_rows:end_rows, start_cols:end_cols, :]

                list_cuboids_local.append(cuboid_data)
                list_cuboids_pixmap_local.append([start_rows,end_rows,start_cols,end_cols])


        array_cuboids = np.array(list_cuboids_local).reshape((frame_size_y - self.CubSizeY) / self.StridesSpace + 1,
                                                             (frame_size_x - self.CubSizeX) / self.StridesSpace + 1,
                                                             self.CubSizeY,
                                                             self.CubSizeX,
                                                             self.NumChannels * self.CubTimesteps)

        relevant_rows_cols_map_cuboids = np.array(list_cuboids_pixmap_local).reshape((frame_size_y - self.CubSizeY) / self.StridesSpace + 1,
                                                                                     (frame_size_x - self.CubSizeX) / self.StridesSpace + 1,
                                                                                      4)

        return array_cuboids,relevant_rows_cols_map_cuboids

class Video_Stream_ARTIF:

    def __init__(self, video_path,video_train_test, size_y,size_x,timesteps,num=-1,ts_first_or_last='first',strides=1,tstrides=1,anompth=0.0,bkgsub=False):

        # Initialize-video-params
        self.video_path = 'INIT_PATH_TO_UCSD'
        self.video_train_test = 'Test'
        self.size_y = 8
        self.size_x = 8
        self.timesteps = 4
        self.frame_size = (128, 128)
        self.data_max_possible = 255.0
        self.seek = -1
        self.list_images_relevant_full_dset = None
        self.list_images_relevant_gt_full_dset = None
        self.seek_dict = {}
        self.seek_dict_gt = {}

        self.list_cuboids = []
        self.list_cuboids_pixmap = []
        self.list_cuboids_anomaly = []
        self.list_cuboids_anompercentage = []

        self.list_all_cuboids = []
        self.list_all_cuboids_gt = []
        self.video_path = video_path
        self.video_train_test = video_train_test
        self.size_y = size_y
        self.size_x = size_x
        self.timesteps = timesteps
        self.ts_first_or_last = ts_first_or_last
        self.list_images_relevant_full_dset, self.list_images_relevant_gt_full_dset = make_file_list(video_path, train_test=video_train_test,n_frames=timesteps,num=num,tstrides=tstrides)
        self.strides = strides
        self.anompth = anompth
        self.bkgsub = bkgsub
        i = 0
        j = 0
        k = 0

        while (i<len(self.list_images_relevant_full_dset)):

            list_images = self.list_images_relevant_full_dset[i]
            list_images_gt = self.list_images_relevant_gt_full_dset[i]

            j=0

            lpre = list_images[j:j + timesteps]
            lcurrent = list_images[j + 1:j + 1 + timesteps]
            lpost = list_images[j + 2:j + 2 + timesteps]

            self.seek_dict[k] = [lpre, lcurrent, lpost]

            if (self.video_train_test == 'Test'):

                lpre_gt = list_images_gt[j:j + timesteps]
                lcurrent_gt = list_images_gt[j + 1:j + 1 + timesteps]
                lpost_gt = list_images_gt[j + 2:j + 2 + timesteps]

                self.seek_dict_gt[k] = [lpre_gt, lcurrent_gt, lpost_gt]
                k += 1
            else:
                self.seek_dict_gt[k] = [None]
                k += 1

            j += 3

            while(j+timesteps<=len(list_images)):

                lpost = list_images[j:j+timesteps]

                self.seek_dict[k] = [lpost]

                if(self.video_train_test=='Test'):

                    lpost_gt = list_images_gt[j:j+timesteps]

                    self.seek_dict_gt[k] = [lpost_gt]
                    k += 1
                else:
                    self.seek_dict_gt[k] = [None]
                    k += 1

                j+=1


            i+=1

    def get_next_cuboids_from_stream(self,gs=False):
        self.seek+=1

        if(self.seek>=len(self.seek_dict)):
            self.seek=-1
            return np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0])
        else:

            list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly, list_all_cuboids, list_all_cuboids_gt,list_cuboids_anompercentage = \
            make_cuboids_for_stream(self,self.seek_dict[self.seek], self.seek_dict_gt[self.seek], self.size_x, self.size_y,
                                    test_or_train=self.video_train_test, ts_first_last = self.ts_first_or_last,strides=self.strides,gs=gs,anompth=self.anompth,
                                    bkgsub=self.bkgsub)

            del (self.list_cuboids[:])
            self.list_cuboids = copy.copy(list_cuboids)
            del(self.list_cuboids_pixmap[:])
            self.list_cuboids_pixmap = copy.copy(list_cuboids_pixmap)
            del(self.list_cuboids_anomaly[:])
            self.list_cuboids_anomaly = copy.copy(list_cuboids_anomaly)

            del(self.list_cuboids_anompercentage[:])
            self.list_cuboids_anompercentage = copy.copy(list_cuboids_anompercentage)

            del(self.list_all_cuboids[:])
            self.list_all_cuboids = copy.copy(list_all_cuboids)
            del(self.list_all_cuboids_gt[:])
            self.list_all_cuboids_gt = copy.copy(list_all_cuboids_gt)

        return list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly, np.array(list_all_cuboids), np.array(list_all_cuboids_gt), list_cuboids_anompercentage

    def reset_stream(self):
        self.seek = -1

    def set_seek_to_video(self,vid):

        if (vid == 0):
            return True

        i=-1
        self.seek+=1

        while i<vid:

            if (len(self.seek_dict[self.seek]) > 1):
                i+=1
            self.seek+=1

        self.seek-=2

        return True

class Format_York_Videos:

    def __init__(self,path,start_anomaly):

        self.path = path
        self.frames_path = os.path.join(path,'frames')
        self.GT_frames_path = os.path.join(path,'GT')

        frames_list = os.listdir(self.frames_path)
        GT_frames_list = os.listdir(self.GT_frames_path)

        frames_list.sort()
        GT_frames_list.sort()

        self.frames_list = [os.path.join(self.frames_path,i) for i in frames_list]
        self.GT_frames_list = [os.path.join(self.GT_frames_path, i) for i in GT_frames_list]

        self.end_train = start_anomaly
        self.start_anomaly = start_anomaly

        self.make_folder_structure()
        self.get_image_shape()
        self.copy_to_train()
        self.copy_eval_test_files()

    def create_dir(self,path):
        if (not os.path.exists(path)):
            os.makedirs(path)

        return True


    def make_folder_structure(self):
        self.create_dir(os.path.join(self.path,'Train'))
        self.create_dir(os.path.join(self.path, 'Test'))

        self.create_dir(os.path.join(self.path, 'Train','Train001'))
        self.create_dir(os.path.join(self.path, 'Test','Test001'))
        self.create_dir(os.path.join(self.path, 'Test', 'Test001_gt'))

        self.train_frames_path = os.path.join(self.path, 'Train','Train001')
        self.test_frames_path = os.path.join(self.path, 'Test','Test001')
        self.test_frames_gt_path = os.path.join(self.path, 'Test', 'Test001_gt')

        return True

    def get_image_shape(self):

        filename = self.frames_list[0]
        image = imread(filename)
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.image_channels = image.shape[2]

        message_print('IMAGE_HEIGHT'+str(self.image_height)+' IMAGE_WIDTH'+str(self.image_width)+' IMAGE_CHANNELSS'+str(self.image_channels))

        return True

    def copy_to_train(self):

        for i in range(0,self.end_train-1):
            shutil.copy(src=self.frames_list[i],dst=os.path.join(self.train_frames_path,os.path.split(self.frames_list[i])[1]))

        return True

    def copy_eval_test_files(self):

        test_images_start = int(os.path.split(self.GT_frames_list[0])[1].split('-')[1].split('.')[0])
        test_images_end = int(os.path.split(self.GT_frames_list[-1])[1].split('-')[1].split('.')[0])

        for i in range(test_images_start-1, test_images_end):
            shutil.copy(src=self.frames_list[i],
                        dst=os.path.join(self.test_frames_path, os.path.split(self.frames_list[i])[1]))

        for idx,i in enumerate(self.GT_frames_list):

            image = imread(i)
            image_resized = resize(image,output_shape=(self.image_height,self.image_width))
            image_resized = (image_resized > 0.0)

            imsave(fname=os.path.join(self.test_frames_gt_path, os.path.split(i)[1]),arr=image_resized)

        return True

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

def make_cuboids_for_stream(stream,list_images,list_images_gt,size_x,size_y,test_or_train='Train', ts_first_last = 'first',strides=1,gs=True,anompth=0.0,bkgsub=False):


    list_cuboids = deque(stream.list_cuboids)
    list_cuboids_pixmap = deque(stream.list_cuboids_pixmap)
    list_cuboids_anomaly = deque(stream.list_cuboids_anomaly)
    list_cuboids_anompercentage = deque(stream.list_cuboids_anompercentage)

    list_all_cuboids = deque(stream.list_all_cuboids)
    list_all_cuboids_gt = deque(stream.list_all_cuboids_gt)

    if(len(list_images)>1):
        start = True
    else:
        start = False

    for i in range(0,len(list_images)):

        list_cuboids_local = []
        list_cuboids_pixmap_local = []
        list_cuboids_anomaly_local = []
        list_cuboids_anompercentage_local = []

        local_collection = imread_collection(list_images[i],as_grey=gs)

        n_frames = len(local_collection)
        frame_size = local_collection[0].shape


        if(test_or_train=='Test'):
            local_collection_gt = imread_collection(list_images_gt[i],as_grey=True)
            n_channels_gt = 1

        if(bkgsub):
            #Subtract mean
            local_collection = local_collection - np.mean(local_collection,axis=0)

            #Rescale data to 0-1 if gs and 0-255 if not-gs
            local_collection = (local_collection - np.min(local_collection))
            local_collection = local_collection/np.max(local_collection)

            if(not gs):
                local_collection = np.uint8(local_collection*255.0)

        frame_size_y = frame_size[0]
        frame_size_x = frame_size[1]


        if (len(frame_size) == 3):
            n_channels = frame_size[2]
        else:
            n_channels = 1

        assert(ts_first_last=='first' or ts_first_last=='last')

        if (ts_first_last == 'first'):
            lc = np.zeros((n_frames, frame_size_y, frame_size_x, n_channels))

            if (test_or_train == 'Test'):
                lcgt = np.zeros((n_frames, frame_size_y, frame_size_x, n_channels_gt))

            for l in range(0, len(local_collection)):
                lc[l] = local_collection[l].reshape(frame_size_y, frame_size_x, n_channels)
                if (test_or_train == 'Test'):
                    lcgt[l] = local_collection_gt[l].reshape(frame_size_y, frame_size_x, n_channels_gt)

            del local_collection
            local_collection = lc

            if (test_or_train == 'Test'):
                del local_collection_gt
                local_collection_gt = lcgt

        elif (ts_first_last == 'last'):

            lc = np.zeros((frame_size_y, frame_size_x, n_channels*n_frames))

            if (test_or_train == 'Test'):
                lcgt = np.zeros((frame_size_y, frame_size_x, n_channels_gt*n_frames))

            for l in range(0,len(local_collection)):
                lc[:,:,l*n_channels:(l+1)*n_channels] = local_collection[l].reshape(frame_size_y, frame_size_x, n_channels)


            if (test_or_train == 'Test'):
                for l in range(0, len(local_collection_gt)):
                    lcgt[:, :, l * n_channels_gt:(l + 1) * n_channels_gt] = local_collection_gt[l].reshape(frame_size_y,frame_size_x, n_channels_gt)


            del local_collection
            local_collection = lc

            if (test_or_train == 'Test'):
                del local_collection_gt
                local_collection_gt = lcgt


        for j in range(0, frame_size_y,strides):
            for k in range(0, frame_size_x,strides):
                start_rows = int(j - (size_y/2))
                end_rows   = int(j + (size_y/2))

                start_cols = int(k - (size_x/2))
                end_cols   = int(k + (size_x/2))


                if (start_rows < 0 or end_rows > frame_size_y or start_cols < 0 or end_cols > frame_size_x):
                    continue

                if(ts_first_last=='first'):
                    cuboid_data = local_collection[:, start_rows:end_rows, start_cols:end_cols, :]
                elif(ts_first_last=='last'):
                    cuboid_data = local_collection[start_rows:end_rows, start_cols:end_cols, :]

                anomaly_gt = False
                anompercentage = 0.0

                if(test_or_train=='Test'):

                    if (ts_first_last == 'first'):
                        anomaly_gt_sum = np.sum(local_collection_gt[:, start_rows:end_rows, start_cols:end_cols, :])
                    elif (ts_first_last == 'last'):
                        anomaly_gt_sum = np.sum(local_collection_gt[start_rows:end_rows, start_cols:end_cols, :])

                    anompercentage = anomaly_gt_sum/((end_cols-start_cols)*(end_rows-start_rows)*n_channels_gt*n_frames)

                    if(anompercentage>0.0):
                        list_anom_percentage.append(anompercentage)

                    if(anompercentage > anompth):
                        print "ANOMALY DETECTED WITH PERCENTAGE > ", anompth
                        anomaly_gt=True


                list_cuboids_local.append(cuboid_data)

                if(start):
                    list_all_cuboids.append(cuboid_data)
                else:
                    list_all_cuboids.popleft()
                    list_all_cuboids.append(cuboid_data)

                list_cuboids_pixmap_local.append((j,k))
                list_cuboids_anomaly_local.append(anomaly_gt)
                list_cuboids_anompercentage_local.append(anompercentage)

                if(start):
                    list_all_cuboids_gt.append(anomaly_gt)
                else:
                    list_all_cuboids_gt.popleft()
                    list_all_cuboids_gt.append(anomaly_gt)

        if (ts_first_last == 'first'):
            array_cuboids = np.array(list_cuboids_local).reshape((frame_size_y - size_y)/strides + 1, (frame_size_x - size_x)/strides + 1,
                                                                 n_frames, size_y, size_x, n_channels)
        elif (ts_first_last == 'last'):

            array_cuboids = np.array(list_cuboids_local).reshape((frame_size_y - size_y)/strides + 1, (frame_size_x - size_x)/strides + 1,
                                                                 size_y, size_x, n_channels*n_frames)

        pixmap_cuboids = np.array(list_cuboids_pixmap_local).reshape((frame_size_y - size_y)/strides + 1, (frame_size_x - size_x)/strides + 1,2)
        anomaly_gt_cuboids = np.array(list_cuboids_anomaly_local).reshape((frame_size_y - size_y)/strides + 1, (frame_size_x - size_x)/strides + 1)
        anompercentage_cuboids = np.array(list_cuboids_anompercentage_local).reshape((frame_size_y - size_y)/strides + 1, (frame_size_x - size_x)/strides + 1)

        if(start):
            list_cuboids.append(array_cuboids)
            list_cuboids_pixmap.append(pixmap_cuboids)
            list_cuboids_anomaly.append(anomaly_gt_cuboids)
            list_cuboids_anompercentage.append(anompercentage_cuboids)
        else:
            list_cuboids.popleft()
            list_cuboids.append(array_cuboids)

            list_cuboids_pixmap.popleft()
            list_cuboids_pixmap.append(pixmap_cuboids)

            list_cuboids_anomaly.popleft()
            list_cuboids_anomaly.append(anomaly_gt_cuboids)

            list_cuboids_anompercentage.popleft()
            list_cuboids_anompercentage.append(anompercentage_cuboids)


    return list(list_cuboids), list(list_cuboids_pixmap), list(list_cuboids_anomaly), list(list_all_cuboids), list(list_all_cuboids_gt), list(list_cuboids_anompercentage)

def imread_collection(list_images,as_grey=False):

    list_of_read_images = []

    for theta in list_images:
        list_of_read_images.append(imread(theta,as_grey=as_grey))

    return np.array(list_of_read_images)

def make_cuboids_from_frames(loc_of_frames, n_frames, size_x, size_y,test_or_train='Train'):


    assert (n_frames >= 5)  # Frame size have to be greater than 5 for any valuable temporal aspect

    list_images = os.listdir(loc_of_frames)
    list_images.sort()


    list_images_relevant = [os.path.join(loc_of_frames, i) for i in strip_sth(list_images, strip_tag='Store')]

    if(test_or_train=='Test'):
        list_images_gt = os.listdir(loc_of_frames+'_gt')
        list_images_gt.sort()
        list_images_relevant_gt = [os.path.join(loc_of_frames+'_gt', i) for i in strip_sth(list_images_gt, strip_tag='Store')]


    image_collection = imread_collection(list_images_relevant)

    if(test_or_train=='Test'):
        image_collection_gt = imread_collection(list_images_relevant_gt)

    frame_size = image_collection[0].shape

    frame_size_y = frame_size[0]
    frame_size_x = frame_size[1]

    if (len(frame_size) == 3):
        n_channels = frame_size[2]
    else:
        n_channels = 1

    one_side_len = int(n_frames / 2)

    start = one_side_len
    end = len(image_collection) - one_side_len

    list_cuboids = []
    list_cuboids_pixmap = []
    list_cuboids_anomaly = []


    for i in tqdm(range(start, end)):

        local_collection = np.zeros((n_frames,frame_size_y, frame_size_x, n_channels))
        local_collection_gt = np.zeros((n_frames,frame_size_y, frame_size_x, n_channels))
        list_cuboids_local = []
        list_cuboids_pixmap_local = []
        list_cuboids_anomaly_local = []

        for j, k in enumerate(range(i - one_side_len, i + one_side_len + 1)):
            img_reshaped = image_collection[k].reshape(frame_size_y, frame_size_x, n_channels)

            local_collection[j, :, :, :] = img_reshaped

            if(test_or_train=='Test'):
                img_reshaped_gt = image_collection_gt[k].reshape(frame_size_y, frame_size_x, n_channels)

                local_collection_gt[j, :, :, :] = img_reshaped_gt

        for j in range(0, frame_size_y):
            for k in range(0, frame_size_x):

                start_rows = j - (size_y/2)
                end_rows   = j + (size_y/2)

                start_cols = k - (size_x/2)
                end_cols   = k + (size_x/2)


                if (start_rows < 0 or end_rows > frame_size_y or start_cols < 0 or end_cols > frame_size_x):
                    continue

                cuboid_data = local_collection[:, start_rows:end_rows, start_cols:end_cols, :]

                anomaly_gt = False
                if(test_or_train=='Test'):
                    anomaly_gt_sum = np.sum(local_collection_gt[:, start_rows:end_rows, start_cols:end_cols, :])

                    if(anomaly_gt_sum>0):
                        anomaly_gt=True

                assert (cuboid_data.shape == (n_frames,size_y, size_x, n_channels))

                list_cuboids_local.append(cuboid_data)
                list_cuboids_pixmap_local.append((j,k))
                list_cuboids_anomaly_local.append(anomaly_gt)


        array_cuboids = np.array(list_cuboids_local).reshape(frame_size_y-size_y+1, frame_size_x-size_x+1, n_frames,size_y, size_x, n_channels)
        pixmap_cuboids = np.array(list_cuboids_pixmap_local).reshape(frame_size_y-size_y+1, frame_size_x-size_x+1,2)
        anomaly_gt_cuboids = np.array(list_cuboids_anomaly_local).reshape(frame_size_y-size_y+1, frame_size_x-size_x+1)


        list_cuboids.append(array_cuboids)
        list_cuboids_pixmap.append(pixmap_cuboids)
        list_cuboids_anomaly.append(anomaly_gt_cuboids)


    return list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly

def make_cuboids_of_videos(loc_videos, train_test='Train', size_x=11, size_y=11, n_frames=5,num=-1):

    list_dirs = os.listdir(loc_videos)
    list_dirs.sort()
    list_dirs = strip_sth(list_dirs, strip_tag='Store')
    list_dirs = strip_sth(list_dirs, strip_tag='gt')

    if(num!=-1):
        list_dirs=list_dirs[0:num]

    list_cuboids = []
    list_cuboids_pixmap = []
    list_cuboids_anom_gt = []

    for idx, i in enumerate(list_dirs):


        list_cuboids_video, list_cuboids_video_pixmap, list_cuboids_anomaly_gt_video = make_cuboids_from_frames(
                                                                     loc_of_frames=os.path.join(loc_videos, i),
                                                                     n_frames=n_frames,
                                                                     size_x=size_x,
                                                                     size_y=size_y,
                                                                     test_or_train=train_test)

        list_cuboids.append(list_cuboids_video)
        list_cuboids_pixmap.append(list_cuboids_video_pixmap)
        list_cuboids_anom_gt.append(list_cuboids_anomaly_gt_video)


    return list_cuboids, list_cuboids_pixmap, list_cuboids_anom_gt

def make_file_list(loc_videos, train_test='Train',n_frames=5,num=-1,tstrides=1):

    list_dirs = os.listdir(loc_videos)
    list_dirs.sort()
    list_dirs = strip_sth(list_dirs, strip_tag='Store')
    list_dirs = strip_sth(list_dirs, strip_tag='gt')
    list_dirs = strip_sth(list_dirs,strip_tag='Videos')

    if(num!=-1):
        list_dirs=list_dirs[0:num]

    list_images_relevant_full_dset = []
    list_images_relevant_gt_full_dset = []

    for idx, i in enumerate(list_dirs):

        list_images_relevant, list_images_relevant_gt = make_frame_list_(loc_of_frames=os.path.join(loc_videos, i),
                                                                         n_frames=n_frames,
                                                                         test_or_train=train_test,tstrides=tstrides)

        list_images_relevant_full_dset.append(list_images_relevant)
        list_images_relevant_gt_full_dset.append(list_images_relevant_gt)

    return list_images_relevant_full_dset, list_images_relevant_gt_full_dset

def make_frame_list_(loc_of_frames, n_frames,test_or_train='Train',tstrides=1):


    assert (n_frames >= 4)  # Frame size have to be greater than 5 for any valuable temporal aspect

    list_images = os.listdir(loc_of_frames)
    list_images.sort()
    list_images = list_images[0::tstrides]


    list_images_relevant = [os.path.join(loc_of_frames, i) for i in strip_sth(list_images, strip_tag='Store')]
    list_images_relevant_gt = None

    if(test_or_train=='Test'):
        list_images_gt = os.listdir(loc_of_frames+'_gt')
        list_images_gt.sort()
        list_images_gt = list_images_gt[0::tstrides]
        list_images_relevant_gt = [os.path.join(loc_of_frames+'_gt', i) for i in strip_sth(list_images_gt, strip_tag='Store')]

    return list_images_relevant, list_images_relevant_gt

def make_list_test_img_class_for_single_video(loc_of_frames,n_frames=5,tstrides=1):

    assert (n_frames >= 4)  # Frame size have to be greater than 5 for any valuable temporal aspect

    list_images = os.listdir(loc_of_frames)
    list_images.sort()
    list_images = list_images[0::tstrides]

    list_images_relevant = [os.path.join(loc_of_frames, i) for i in strip_sth(list_images, strip_tag='Store')]

    ListTestImgClass_Video = []

    for i in range(0,len(list_images_relevant)-n_frames-1):
        ListTestImgClass_Video.append(TestImgClass(fnames=list_images_relevant[i:i+n_frames+2]))

    return ListTestImgClass_Video

def make_list_test_img_class_all_videos(loc_videos,n_frames=5,tstrides=1,num=-1):

    list_dirs = os.listdir(loc_videos)
    list_dirs.sort()
    list_dirs = strip_sth(list_dirs, strip_tag='Store')
    list_dirs = strip_sth(list_dirs, strip_tag='gt')
    list_dirs = strip_sth(list_dirs, strip_tag='Videos')

    if (num != -1):
        list_dirs = list_dirs[0:num]

    ListTestImgClass_AllVideos = []


    for idx, i in enumerate(list_dirs):
        ListTestImgClass_Video = make_list_test_img_class_for_single_video(loc_of_frames=os.path.join(loc_videos, i),
                                                                           n_frames=n_frames,
                                                                           tstrides=tstrides)

        ListTestImgClass_AllVideos.append(ListTestImgClass_Video)


    return ListTestImgClass_AllVideos

def get_frame_size(test_img,gs):

    list_files_past, list_files_present, list_files_future = test_img.get_time_base_filenames()

    image_collection = imread_collection(list_files_present, as_grey=gs)

    frame_size = image_collection[0].shape
    frame_size_y = frame_size[0]
    frame_size_x = frame_size[1]

    return frame_size_y, frame_size_x

def make_videos_of_anomalies(list_cuboids_test,path_videos,n_frames,size_x,size_y,threshold,cnn=False):

    list_dirs = os.listdir(path_videos)
    list_dirs.sort()
    list_dirs = strip_sth(list_dirs, strip_tag='Store')
    list_dirs = strip_sth(list_dirs, strip_tag='gt')

    list_videos = []

    for idx, i in enumerate(list_dirs):

        list_videos.append(make_anomaly_video_of_one_video(list_cuboids_test[idx], n_frames, size_x,size_y,threshold,cnn))

    return list_videos

def make_anomaly_video_of_one_video(cuboid_array_list, n_frames, size_x,size_y,threshold,cnn=False):

    img_shape=cuboid_array_list[0][0,0].frame_size
    total_number_of_images = len(cuboid_array_list)+ n_frames - 1

    image_collection_true = np.zeros((total_number_of_images,img_shape[0],img_shape[1]))
    image_collection_anom_detected = np.zeros((total_number_of_images,img_shape[0],img_shape[1]))

    rows = cuboid_array_list[0].shape[0]
    cols = cuboid_array_list[0].shape[1]

    for l in range(0, len(cuboid_array_list)):
        for j in range(0, rows):
            for k in range(0, cols):

                cuboid = cuboid_array_list[l][j, k]

                start_frame = cuboid.frame_id - int((n_frames) / 2)
                end_frame = cuboid.frame_id + int((n_frames) / 2) + 1

                if(cnn):
                    start_rows = cuboid.y_centroid - size_y / 2
                    end_rows = cuboid.y_centroid + size_y / 2
                    start_cols = cuboid.x_centroid - size_x / 2
                    end_cols = cuboid.x_centroid + size_x / 2
                else:
                    start_rows = cuboid.y_centroid - size_y / 2
                    end_rows = cuboid.y_centroid + size_y / 2 + 1
                    start_cols = cuboid.x_centroid - size_x / 2
                    end_cols = cuboid.x_centroid + size_x / 2 + 1


                reshaped_data = np.zeros((n_frames,size_x,size_y))
                nchannels=1

                for z in range(0,n_frames):
                    reshaped_data[z,:,:] = cuboid.data[:,:,0,z]

                image_collection_true[start_frame:end_frame, start_rows:end_rows, start_cols:end_cols] = reshaped_data

                if(cuboid_array_list[l][j,k].anom_score>=threshold):
                    image_collection_anom_detected[start_frame:end_frame, start_rows:end_rows,start_cols:end_cols] = np.ones((n_frames, size_x, size_y))


    return [image_collection_true , image_collection_anom_detected]

def make_anomaly_frames(list_videos,local=False,threshold=1.0):

    list_dirs = []

    for idx, i in enumerate(list_videos):

        directory = os.path.join('Results', 'Video_' + str(idx))

        if not os.path.exists(directory):
            os.makedirs(directory)

        images = i[0]
        mask = i[1]

        for idx_image in range(0, len(images)):

            img = img_as_float(np.uint8(images[idx_image]))
            img_color = np.dstack((img, img, img))
            img_hsv = color.rgb2hsv(img_color)

            color_mask = np.zeros((img.shape[0], img.shape[1], 3))
            color_mask[:, :, 0] = np.uint8(mask[idx_image])
            color_mask_hsv = color.rgb2hsv(color_mask)

            img_hsv[..., 0] = color_mask_hsv[..., 0]
            img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.6

            img_masked = color.hsv2rgb(img_hsv)
            plt.imshow(img_masked)

            if(local):
                plt.savefig(os.path.join(directory, 'image_' + str(idx_image).zfill(3) + '_local_.png'))

            else:
                plt.savefig(os.path.join(directory, 'image_' + str(threshold)+'_'+ str(idx_image).zfill(3) + '.png'))

            plt.close()

        list_dirs.append(os.path.join(os.getcwd(), directory))

    return os.path.join(os.getcwd(),'Results'), list_dirs

def make_videos_of_frames(path_results,list_dirs_of_videos,local=False,threshold=1.5):

    dir = os.path.join(path_results,'Videos')

    if not os.path.exists(dir):
        os.makedirs(dir)

    for idx,i in enumerate(list_dirs_of_videos):

        if(local):
            str_command = 'avconv -r 12 -y -i '+ os.path.join(i,'image_%03d_local_.png') + ' ' + os.path.join(dir,str(idx)+'_local.mp4')
        else:
            str_command = 'avconv -r 12 -y -i ' + os.path.join(i, 'image_'+str(threshold)+'_'+'%03d.png') + ' ' + os.path.join(dir, str(threshold)+'_'+str(idx) + '.mp4')

        os.system(str_command)

    return True

def mean_squared_error(x,y):
    return np.sqrt(np.mean((x-y)**2))
