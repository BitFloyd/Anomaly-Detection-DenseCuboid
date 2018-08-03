import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import keras_contrib.backend as KC
import imageio
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Lambda, Reshape, GaussianNoise, Conv2D, SpatialDropout2D, LeakyReLU
from keras.layers import Dropout, MaxPooling2D, Layer, merge, AveragePooling2D
from keras.layers import UpSampling2D, Flatten, BatchNormalization,ConvLSTM2D,TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.losses import mean_squared_error,binary_crossentropy
from keras.utils import multi_gpu_model
from keras_contrib.losses import DSSIMObjective
import os
import time
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler,Normalizer
import keras.backend as K
from keras.objectives import *
import h5py
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from functionals_pkg import save_objects as so
from functionals_pkg.logging import  message_print,debug_print
from sklearn.mixture import GaussianMixture
import seaborn as sns
import itertools
import multiprocessing
sns.set(color_codes=True)
import thread


def input_thread(a_list):
    string = raw_input()
    if(string == 'stop'):
        a_list.append(True)


class CustomClusterLayer(Layer):

    def __init__(self,loss_fn,means,lamda,n_clusters,hid,reverse=False,**kwargs):

        self.is_placeholder = True
        self.loss_fn=loss_fn
        self.reverse = reverse

        self.means=K.variable(value=K.cast_to_floatx(means),name='means_tensor_in')
        self.lamda=K.variable(value=K.cast_to_floatx(lamda),name='lamda_tensor_in')

        self.n_clusters = n_clusters
        self.hid = hid
        self.cl_loss_wt = K.variable(value=K.cast_to_floatx(x=0.0),name='cl_loss_wt_in_model')

        super(CustomClusterLayer, self).__init__(**kwargs)

    def custom_loss_clustering(self,x_true, x_pred, encoded_feats):

        if(self.reverse):
            loss = K.mean(self.loss_fn(K.reverse(x_true,axes=3), x_pred))
        else:
            loss = K.mean(self.loss_fn(x_true, x_pred))

        centroids = self.means

        M = self.n_clusters
        N = K.shape(x_true)[0]

        rep_centroids = K.reshape(K.tile(centroids, [N, 1]), [N, M, self.hid])
        rep_points = K.reshape(K.tile(encoded_feats, [1, M]), [N, M, self.hid])
        sum_squares = K.tf.reduce_sum(K.square(rep_points - rep_centroids),reduction_indices=2)
        best_centroids = K.argmin(sum_squares, 1)

        cl_loss = self.cl_loss_wt * (self.lamda) * K.mean(K.sqrt(K.sum(K.square(encoded_feats-K.gather(centroids,best_centroids)),axis=1)))

        return loss+cl_loss

    def call(self, inputs):

        x_true  = inputs[0]
        x_pred  = inputs[1]
        encoded_feats = inputs[2]

        loss = self.custom_loss_clustering(x_true, x_pred,encoded_feats)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x_pred

class CustomClusterLayer_Test(Layer):

    def __init__(self,loss_fn,means,lamda,n_clusters,hid,assignment_lamda,reverse=False,**kwargs):

        self.is_placeholder = True
        self.loss_fn=loss_fn
        self.assignment_lamda = assignment_lamda
        self.n_clusters = n_clusters
        self.hid = hid
        self.reverse = reverse
        self.means=K.variable(value=K.cast_to_floatx(means),name='means_tensor_in')
        self.lamda=K.variable(value=K.cast_to_floatx(lamda),name='lamda_tensor_in')


        self.cl_loss_wt = K.variable(value=K.cast_to_floatx(x=0.0),name='cl_loss_wt_in_model')

        super(CustomClusterLayer_Test, self).__init__(**kwargs)

    def custom_loss_clustering(self,x_true, x_pred, encoded_feats,assignments):

        if(self.reverse):
            loss = K.mean(self.loss_fn(K.reverse(x_true,axes=0), x_pred))
        else:
            loss = K.mean(self.loss_fn(x_true, x_pred))

        centroids = self.means

        M = self.n_clusters
        N = K.shape(x_true)[0]

        rep_centroids = K.reshape(K.tile(centroids, [N, 1]), [N, M, self.hid])
        rep_points = K.reshape(K.tile(encoded_feats, [1, M]), [N, M, self.hid])
        sum_squares = K.tf.reduce_sum(K.square(rep_points - rep_centroids),reduction_indices=2)
        best_centroids = K.argmin(sum_squares, 1)


        cl_loss = self.cl_loss_wt * (self.lamda) * K.mean(K.sqrt(K.sum(K.square(encoded_feats-K.gather(centroids,best_centroids)),axis=1)))

        zero = K.constant(0,dtype='int64')
        differences = best_centroids - K.cast(assignments,dtype='int64')
        unequal_assignments = K.not_equal(x=differences,y=zero)
        mean_all_unequal = K.mean(K.cast(unequal_assignments,dtype='float32'))
        assignment_loss = self.cl_loss_wt * (self.assignment_lamda) * mean_all_unequal

        return loss+cl_loss+assignment_loss

    def call(self, inputs):

        x_true        = inputs[0]
        x_pred        = inputs[1]
        encoded_feats = inputs[2]
        assignments   = inputs[3]
        loss = self.custom_loss_clustering(x_true, x_pred,encoded_feats,assignments)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x_pred

class Super_autoencoder:

    def __init__(self, model_store, size_y=32, size_x=32, n_channels=3, h_units=256, n_timesteps=5,
                 loss='mse', batch_size=64, n_clusters=10, lr_model=1e-4, lamda=0.01,
                 gs=False, notrain=False, reverse=False, data_folder='data_store',
                 dat_h5=None,large=True, means_tol=5e-5, means_patience=200, max_fit_tries=500000):

        self.means = None
        self.initial_means = None
        self.list_mean_disp = []
        self.loss_list = []
        self.features_h5 = None
        self.features_h5_pre = None
        self.y_obj = None
        self.ae = None
        self.encoder = None
        self.decoder = None
        self.multi_gpu_model = False
        self.n_gpu = 1

        self.clustering_lr = 1.0
        self.batch_size = batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model
        self.lamda = lamda
        self.h_units = h_units
        self.means = np.zeros((n_clusters, h_units))
        self.cluster_assigns = None
        self.ntsteps = n_timesteps
        self.km = MiniBatchKMeans(n_clusters=self.n_clusters, verbose=0)
        self.dict = DictionaryLearning()
        self.x_train = [100, 10, 10]
        self.gs = gs
        self.notrain = notrain
        self.reverse = reverse
        self.n_channels = n_channels
        self.large = large

        self.means_tol = means_tol
        self.means_patience = means_patience
        self.means_batch = 256
        self.max_fit_tries = max_fit_tries

        self.dat_folder = data_folder
        self.dat_h5 = dat_h5

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

    def set_x_train(self, id):
        del (self.x_train)
        print "Loading Chapter : ", 'chapter_' + str(id) + '.npy'
        self.x_train = np.array(self.dat_h5.get('chapter_' + str(id)))

    def set_feats(self, id):

        print "Loading Chapter Feats : ", 'chapter_' + str(id) + '.npy'
        return (np.array(self.features_h5.get('chapter_' + str(id))))

    def set_feats_pre(self, id):

        print "Loading Chapter Feats : ", 'chapter_' + str(id) + '.npy'
        return (np.array(self.features_h5_pre.get('chapter_' + str(id))))

    def set_cl_loss(self,val):
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(val))
        return True

    def return_all_encodings(self):

        features_h5 = h5py.File(os.path.join(self.model_store, 'features.h5'), 'r')

        list_feats = []

        for id in range(0, len(features_h5)):
            f_arr = np.array(features_h5.get('chapter_' + str(id)))
            list_feats.extend(f_arr.tolist())
            del f_arr

        list_feats = np.array(list_feats)
        features_h5.close()

        return list_feats

    def make_ae_model_multi_gpu(self,n_gpus):

        try:
            self.ae_single = self.ae
            self.ae = multi_gpu_model(self.ae, gpus=n_gpus)
            print("Training using multiple GPUs..")
            self.ae.compile(optimizer=self.adam_ae, loss=None)
            self.multi_gpu_model = True
            self.n_gpu = n_gpus
        except:
            print("Training using single GPU or CPU..")
            self.multi_gpu_model = False
            self.n_gpu = 1

        return True

    def set_clusters_to_optimum(self):

        if(os.path.exists(os.path.join(self.model_store,'nclusters_optimized.pkl'))):
            self.n_clusters = so.load_obj(os.path.join(self.model_store,'nclusters_optimized.pkl'))
        else:
            raise Exception ('No optimized clusters saved in the model')

        return True

    def fit_model_using_datagen(self,generator, verbose=1, num_initial_epochs = 3, earlystopping=False, patience=10, least_loss=1e-5,
                                num_max_epochs=100,reduce_lr = False, patience_lr=5 , factor=1.5):

        if (self.notrain):
            return True

        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        # start_initial chapters training
        n_cpus = multiprocessing.cpu_count()
        message_print("MULTITHREADING IN " + str(n_cpus) + "CPUS")
        history = self.ae.fit_generator(generator=generator,use_multiprocessing=True,workers=n_cpus,epochs=num_initial_epochs,
                                        verbose=verbose,max_queue_size=1000)

        self.loss_list.append(history.history['loss'][0])



        # Get means of predicted features
        feats = self.encoder.predict_generator(generator=generator,use_multiprocessing=True,workers=n_cpus,verbose=verbose)

        message_print("START INITIAL KMEANS FITTING")
        message_print("CLUSTERS :"+str(self.n_clusters))

        del self.km
        self.km = MiniBatchKMeans(n_clusters=self.n_clusters, verbose=1,batch_size=self.means_batch,
                                               compute_labels=False,tol=1e-9,max_no_improvement=self.means_patience*200)

        start = time.time()
        self.km.fit(feats)
        end = time.time()

        print "TIME TAKEN:",(end - start) / 3600.0, "HOURS"

        self.means = np.copy(self.km.cluster_centers_).astype('float64')

        message_print("MEANS_INITIAL")

        print self.means

        self.initial_means = np.copy(self.km.cluster_centers_).astype('float64')

        np.save(os.path.join(self.model_store, 'initial_means.npy'), self.initial_means)

        loss_track = 0
        loss_track_lr = 0
        lowest_loss_ever = 1000.0

        # Make clustering loss weight to 1
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(1.0))
        K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

        lr = self.lr_model

        for j in range(num_initial_epochs, num_max_epochs):

            print (j), "/", num_max_epochs, ":"

            history = self.ae.fit_generator(generator=generator,use_multiprocessing=True,workers=n_cpus,epochs=1,verbose=verbose
                                            ,max_queue_size=1000)
            current_loss = history.history['loss'][0]
            self.loss_list.append(history.history['loss'][0])
            feats = self.encoder.predict_generator(generator=generator,use_multiprocessing=True,workers=n_cpus,verbose=verbose
                                                   ,max_queue_size=1000)
            self.cluster_assigns = self.get_assigns(self.means, feats)
            means_pre = np.copy(self.means)
            self.update_means(feats, self.cluster_assigns)

            # update cluster assigns for the next loop
            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

            K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

            if (lowest_loss_ever - current_loss > least_loss):
                loss_track = 0
                loss_track_lr = 0
                print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                lowest_loss_ever = current_loss

            else:
                print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                loss_track += 1
                loss_track_lr += 1
                print "Loss track is :", loss_track, "/", patience
                print "Loss track lr is :", loss_track_lr, "/", patience_lr

            if (reduce_lr and loss_track_lr > patience_lr):
                loss_track_lr = 0
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "REDUCING_LR AT EPOCH :", j
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                K.set_value(self.ae.optimizer.lr, lr / factor)
                lr = lr / factor
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "REDUCING_LR TO:", lr
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

            if (earlystopping and loss_track > patience):
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "EARLY STOPPING AT EPOCH :", j
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                break

        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        print "PICKLING LISTS AND SAVING WEIGHTS"
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

        with open(os.path.join(self.model_store, 'losslist.pkl'), 'wb') as f:
            pickle.dump(self.loss_list, f)

        with open(os.path.join(self.model_store, 'meandisp.pkl'), 'wb') as f:
            pickle.dump(self.list_mean_disp, f)

        self.save_weights()

        #Create features h5 dataset
        feats = self.encoder.predict_generator(generator=generator, use_multiprocessing=True, workers=6,
                                               verbose=verbose)
        with h5py.File(os.path.join(self.model_store, 'features.h5'), "a") as f:
            dset = f.create_dataset('chapter_' + str(0), data=np.array(feats))
            print(dset.shape)

        np.save(os.path.join(self.model_store, 'means.npy'), self.means)

        return True

    def fit_model_ae_chaps(self, verbose=1, n_initial_chapters=10, earlystopping=False, patience=10, least_loss=1e-5,
                           n_chapters=20,n_train=2,reduce_lr = False, patience_lr=5 , factor=1.5):

        if (self.notrain):
            return True

        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        # start_initial chapters training

        for i in range(0, n_initial_chapters):
            self.set_x_train(i)

            history = self.ae.fit(self.x_train, shuffle=True, epochs=1,
                                   batch_size=self.batch_size, verbose=verbose)

            self.loss_list.append(history.history['loss'][0])


        for i in range(0,n_chapters):
            self.set_x_train(i)
            feats = self.encoder.predict(self.x_train)
            name = 'chapter_' + str(i)
            with h5py.File(os.path.join(self.model_store, 'features_pre.h5'), "a") as f:
                if(name in f.keys()):
                    del f[name]
                dset = f.create_dataset(name, data=np.array(feats))
                print(dset.shape)
            del feats

        fit_tries = 0
        disp_track = 0
        means_patience = self.means_patience
        max_fit_tries = n_chapters*500

        if(n_chapters<5):
            max_fit_tries = n_chapters*5e3

        self.features_h5_pre = h5py.File(os.path.join(self.model_store, 'features_pre.h5'), 'r')

        while (disp_track < means_patience and fit_tries <= max_fit_tries):

            for i in range(0, n_chapters):

                if (disp_track >= means_patience or fit_tries > max_fit_tries):
                    break

                feats = self.set_feats_pre(i)

                if (fit_tries == 0):
                    bef_fit = np.zeros((self.n_clusters,self.h_units))
                else:
                    bef_fit = np.copy(self.km.cluster_centers_).astype('float64')

                try:
                    for k in range(0, len(feats), self.means_batch):
                        if (k + self.means_batch < len(feats)):
                            self.km.partial_fit(feats[k:k + self.means_batch])
                        else:
                            self.km.partial_fit(feats[k:])

                    aft_fit = np.copy(self.km.cluster_centers_).astype('float64')

                    fit_tries += 1

                    disp_means = np.sum(np.linalg.norm(bef_fit - aft_fit, axis=1))

                    if (disp_means < self.means_tol):
                        disp_track += 1
                    else:
                        disp_track = 0
                except:
                    fit_tries += 1
                    message_print("MEANS FIT TRY FAILED. EXCEPTING")

                print "-------------------------"
                print "DISP:", disp_means
                print "DISP_TRACK:", disp_track
                print "FIT_TRIES:", fit_tries
                print "-------------------------"

                del feats


        print "FINISH MEANS INITIAL"
        self.features_h5_pre.close()

        self.means = np.copy(self.km.cluster_centers_).astype('float64')

        print "$$$$$$$$$$$$$$$$$$"
        print "MEANS_INITIAL"
        print "$$$$$$$$$$$$$$$$$$"
        print self.means

        self.initial_means = np.copy(self.km.cluster_centers_)
        np.save(os.path.join(self.model_store, 'initial_means.npy'), self.initial_means)

        loss_track = 0
        loss_track_lr = 0
        lowest_loss_ever = 1000.0

        # Make clustering loss weight to 1
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(1.0))
        K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

        lr = self.lr_model

        total_num_samples = 0
        total_loss = 0.0

        for j in range(n_initial_chapters, n_chapters):

            print (j), "/", n_chapters, ":"
            self.set_x_train(j)

            total_num_samples+=len(self.x_train)

            history = self.ae.fit(self.x_train, batch_size=self.batch_size, epochs=1,
                                  verbose=verbose, shuffle=True)

            total_loss += history.history['loss'][0]*len(self.x_train)

            feats = self.encoder.predict(self.x_train)
            self.cluster_assigns = self.get_assigns(self.means, feats)
            means_pre = np.copy(self.means)
            self.update_means(feats, self.cluster_assigns)

            # update cluster assigns for the next loop
            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

            K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

            del self.means
            del self.cluster_assigns

        if(n_initial_chapters < n_chapters):

            current_loss = total_loss/total_num_samples
            lowest_loss_ever = current_loss
            self.loss_list.append(current_loss)

        if(n_train > 1):

            a_list = []
            thread.start_new_thread(input_thread, (a_list,))

            it_time_mins = 0
            for i in range(1,n_train+1):

                it_start_time = time.time()
                print "#############################"
                print "N_TRAIN: ", i
                print "#############################"

                if (earlystopping and loss_track > patience):
                    break

                total_num_samples = 0
                total_loss = 0.0

                for j in range(0, n_chapters):

                    print (j), "/", n_chapters, ":"
                    self.set_x_train(j)

                    total_num_samples += len(self.x_train)

                    history = self.ae.fit(self.x_train, batch_size=self.batch_size, epochs=1,
                                          verbose=verbose, shuffle=True)

                    total_loss += history.history['loss'][0] * len(self.x_train)

                    print "NTRAIN: ", i, '/', n_train

                    feats = self.encoder.predict(self.x_train)

                    self.cluster_assigns = self.get_assigns(self.means, feats)

                    means_pre = np.copy(self.means)
                    self.update_means(feats, self.cluster_assigns)


                    self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

                    K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

                    del feats
                    del self.cluster_assigns

                current_loss = total_loss / total_num_samples
                self.loss_list.append(current_loss)

                if (lowest_loss_ever - current_loss > least_loss):
                    loss_track = 0
                    loss_track_lr = 0
                    print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                    lowest_loss_ever = current_loss

                else:
                    print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                    loss_track += 1
                    loss_track_lr += 1
                    print "Loss track is :", loss_track, "/", patience
                    print "Loss track lr is :", loss_track_lr, "/", patience_lr

                if (reduce_lr and loss_track_lr > patience_lr):
                    loss_track_lr = 0
                    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    print "REDUCING_LR AT EPOCH :", i
                    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    K.set_value(self.ae.optimizer.lr, lr / factor)
                    lr = lr / factor
                    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    print "REDUCING_LR TO:", lr
                    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

                if (earlystopping and loss_track > patience):
                    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    print "EARLY STOPPING AT EPOCH :", i
                    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    break

                it_end_time = time.time()

                it_time_mins = (it_time_mins*(i)+(it_end_time-it_start_time)/60.0)/(i+1)

                message_print("TIME_TAKEN : " + str(it_time_mins) + " MINUTES")
                message_print("TIME_LEFT : " + str((it_time_mins)*(n_train-i)/60) + "HOURS")

                if a_list:
                    message_print("STOP DETECTED.... STOPPING THE TRAINING")
                    break

        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        print "PICKLING LISTS AND SAVING WEIGHTS"
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

        with open(os.path.join(self.model_store, 'losslist.pkl'), 'wb') as f:
            pickle.dump(self.loss_list, f)

        with open(os.path.join(self.model_store, 'meandisp.pkl'), 'wb') as f:
            pickle.dump(self.list_mean_disp, f)

        self.save_weights()

        #Create features h5 dataset
        for i in range(0,n_chapters):
            self.set_x_train(i)
            feats = self.encoder.predict(self.x_train)
            with h5py.File(os.path.join(self.model_store, 'features.h5'), "a") as f:
                dset = f.create_dataset('chapter_' + str(i), data=np.array(feats))
                print(dset.shape)
            del feats

        np.save(os.path.join(self.model_store, 'means.npy'), self.means)

        return True


    def fit_model_ae_chaps_nocloss(self, verbose=1, earlystopping=False, patience=10, least_loss=1e-5,
                                   n_chapters=20,n_train=2,reduce_lr = False, patience_lr=5 , factor=1.5):

        if (self.notrain):
            return True

        loss_track = 0
        loss_track_lr = 0
        lowest_loss_ever = 1000.0

        lr = self.lr_model

        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        for i in range(0, n_train ):

            print "#############################"
            print "N_TRAIN: ", i
            print "#############################"

            if (earlystopping and loss_track > patience):
                break

            for j in range(0, n_chapters):

                print (j), "/", n_chapters, ":"
                self.set_x_train(j)

                print "NTRAIN: ", i, '/', n_train

                history = self.ae.fit(self.x_train, batch_size=self.batch_size, epochs=1,
                                      verbose=verbose, shuffle=True)

                current_loss = history.history['loss'][0]
                self.loss_list.append(history.history['loss'][0])


                feats = self.encoder.predict(self.x_train)

                self.km.partial_fit(feats)

                del feats

                means_pre = np.copy(self.means)

                self.means = np.copy(self.km.cluster_centers_).astype('float64')

                self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

                K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

                if (len(self.x_train) > 20000):
                    if (lowest_loss_ever - current_loss > least_loss):
                        loss_track = 0
                        loss_track_lr = 0
                        print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                        lowest_loss_ever = current_loss

                    else:
                        print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                        loss_track += 1
                        loss_track_lr += 1
                        print "Loss track is :", loss_track, "/", patience
                        print "Loss track lr is :", loss_track_lr, "/", patience_lr

                    if (reduce_lr and loss_track_lr > patience_lr):
                        loss_track_lr = 0
                        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                        print "REDUCING_LR AT EPOCH :", j
                        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                        K.set_value(self.ae.optimizer.lr, lr / factor)
                        lr = lr / factor
                        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                        print "REDUCING_LR TO:", lr
                        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

                    if (earlystopping and loss_track > patience):
                        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                        print "EARLY STOPPING AT EPOCH :", j
                        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                        break

        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        print "PICKLING LISTS AND SAVING WEIGHTS"
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

        with open(os.path.join(self.model_store, 'losslist.pkl'), 'wb') as f:
            pickle.dump(self.loss_list, f)

        with open(os.path.join(self.model_store, 'meandisp.pkl'), 'wb') as f:
            pickle.dump(self.list_mean_disp, f)

        #Create features h5 dataset
        for i in range(0,n_chapters):
            self.set_x_train(i)
            feats = self.encoder.predict(self.x_train)
            with h5py.File(os.path.join(self.model_store, 'features.h5'), "a") as f:
                if('chapter_' + str(i) in f.keys()):
                    del f['chapter_' + str(i)]
                dset = f.create_dataset('chapter_' + str(i), data=np.array(feats))
                print(dset.shape)
            del feats

        self.save_weights()

        return True

    def perform_feature_space_analysis(self):

        list_feats = self.return_all_encodings()

        pdf_name = os.path.join(self.model_store, 'features_kde.pdf')

        if(os.path.exists(pdf_name)):
            os.remove(pdf_name)

        if(list_feats.shape[1]<=16):
            cols_plots = list_feats.shape[1]
            rows_plots = 1
        else:
            cols_plots = 16
            rows_plots = (list_feats.shape[1] / cols_plots)

        with PdfPages(pdf_name) as pdf:

            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            fig, ax = plt.subplots(ncols=cols_plots, nrows=rows_plots, figsize=(200, 200), sharex='col', sharey='row')
            list_feats_to_plot = list_feats
            plt.suptitle('Distribution of each feature in the dataset',fontsize=30)

            for i in range(0, list_feats_to_plot.shape[1]):
                print "PROCESSING FEATURE:", i

                if(list_feats.shape[1]<=16):
                    ax_object = ax[int(i % cols_plots)]
                else:
                    ax_object = ax[int(i / cols_plots)][i % cols_plots]

                ax_object.set_title('feature: ' + str(i + 1),fontsize=20)
                plt.setp(ax_object.get_xticklabels(),visible=True)
                try:
                    sns.distplot(list_feats_to_plot[:, i], kde=True, rug=False, hist=True,
                                 ax=ax_object)
                except:
                    print "SKIPPING FEATURE:", i

            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    def check_model_sanity(self):

        self.set_x_train(0)

        with open(os.path.join(self.model_store,'evaluation_score.txt')) as f:
            content = f.read().splitlines()

        score_on_chap_0_saved_list = [float(i.split('=>')[-1]) for i in content]

        score_on_chap_0 = self.ae.evaluate(self.x_train)

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "scores_saved = ", score_on_chap_0_saved_list
        print "score_after_load = ", score_on_chap_0
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

        f = open(os.path.join(self.model_store, 'evaluation_score.txt'), 'a+')
        f.write('score_evaluated_chap0_after_load: => ' + str(score_on_chap_0) + '\n')
        f.close()

        encoded_feats_post_load = self.encoder.predict(self.x_train)
        encoded_feats_pre_load = np.load(os.path.join(self.model_store, 'encoded_feats_sanity.npy'))

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "difference_in_encoded = ", np.sum(np.abs(encoded_feats_post_load-encoded_feats_pre_load))
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"


        decodeds_post_load = self.decoder.predict(encoded_feats_post_load)
        decoded_feats_pre_load = np.load(os.path.join(self.model_store, 'decoded_feats_sanity.npy'))

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "difference_in_decoded = ", np.sum(np.abs(decodeds_post_load-decoded_feats_pre_load))
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

        return True

    def prepare_for_model_sanity(self):

        self.set_x_train(0)

        for i in range(0,10):

            score_on_chap_0 = self.ae.evaluate(self.x_train)

            print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
            print "score_saved = ", score_on_chap_0
            print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

            f = open(os.path.join(self.model_store, 'evaluation_score.txt'), 'a+')
            f.write('score_evaluated_chap0: '+str(i) + ' => ' + str(score_on_chap_0) + '\n')
            f.close()

        encoded_feats = self.encoder.predict(self.x_train)
        np.save(os.path.join(self.model_store, 'encoded_feats_sanity.npy'), encoded_feats)

        decodeds = self.decoder.predict(encoded_feats)
        np.save(os.path.join(self.model_store, 'decoded_feats_sanity.npy'), decodeds)

    def perform_kmeans(self,partial=False):

        list_feats = shuffle(self.return_all_encodings())

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "START KMEANS FITTING"
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        message_print("CLUSTERS :"+str(self.n_clusters))

        del self.km
        if(partial):
            self.km = MiniBatchKMeans(n_clusters=self.n_clusters, verbose=1,batch_size=self.means_batch,
                                               compute_labels=False,tol=1e-9,max_no_improvement=self.means_patience*200)
        else:
            self.km = KMeans(n_clusters=self.n_clusters, max_iter=int(1e3), verbose=1, n_jobs=-1)

        start = time.time()
        self.km.fit(list_feats)
        end = time.time()

        print "TIME TAKEN:",(end - start) / 3600.0, "HOURS"

        self.means = np.copy(self.km.cluster_centers_).astype('float64')
        np.save(os.path.join(self.model_store, 'means.npy'), self.means)


        pickle.dump(self.km, open('kmeans_obj.pkl', 'wb'))

        return True

    def perform_dict_learn(self,guill=False,n_comp=64):

        list_feats = shuffle(self.return_all_encodings())

        if(n_comp<1):
            n_components = list_feats.shape[1]
        else:
            n_components = n_comp

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "START DICTIONARY FITTING"
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

        print "N_COMPONENTS =",n_components

        if(guill):
            print "RUNNING IN GUILLIMIN MODE"
            self.dict  = MiniBatchDictionaryLearning(n_components=n_components,verbose=1,n_iter=1000,batch_size=40)
        else:
            self.dict = MiniBatchDictionaryLearning(n_components=n_components, verbose=1, n_jobs=-1,n_iter=1000,batch_size=40)

        start = time.time()
        self.dict.fit(list_feats)
        end = time.time()

        print "TIME TAKEN:",(end - start) / 3600.0, "HOURS"

        print "N_BASIS_COMPONENTS:", self.dict.components_.shape[0]

        if(os.path.exists(os.path.join(self.model_store,'basis_dict.pkl'))):
            os.remove(os.path.join(self.model_store,'basis_dict.pkl'))

        so.save_obj(self.dict,os.path.join(self.model_store,'basis_dict.pkl'))

        return True

    def perform_gmm_training(self,guill=False,n_comp=-1,covariance_type='full'):

        list_feats = shuffle(self.return_all_encodings())

        if(n_comp == -1):
            n_comp = self.n_clusters

        self.gm = GaussianMixture(n_components=n_comp,max_iter=int(1e3),n_init=10,verbose=1,verbose_interval=100,covariance_type=covariance_type)

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "START GMM FITTING: NUMBER OF COMPONENTS = ", n_comp
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

        start = time.time()
        self.gm.fit(list_feats)
        end = time.time()

        print "TIME TAKEN:", (end - start) / 3600.0, "HOURS"

        if (os.path.exists(os.path.join(self.model_store, 'gmm.pkl'))):
            os.remove(os.path.join(self.model_store, 'gmm.pkl'))

        so.save_obj(self.gm,os.path.join(self.model_store, 'gmm.pkl'))

        return  True

    def perform_gmm_analysis_and_training(self,guill=False):

        list_feats = shuffle(self.return_all_encodings())

        lowest_bic = np.infty
        bic = []
        n_components_range = range(5, 31)

        print "N_COMPONENTS TO BE ANALYZED FOR GAUSSIAN MIXTURE:", n_components_range

        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type, verbose=1)
                start_time = time.time()
                gmm.fit(list_feats)
                end_time = time.time()
                bic.append(gmm.bic(list_feats))
                print "---------------------------------------------------"
                print "cv_type:",cv_type," ","n_components:",n_components
                print "---------------------------------------------------"
                print "bic:",bic[-1]
                print "---------------------------------------------------"
                print "time:",(end_time-start_time)/3600.0
                print "---------------------------------------------------"
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_n_components = n_components
                    best_cv_type = cv_type



        bic = np.array(bic)
        color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange'])
        bars = []

        # Plot the BIC scores
        spl = plt.subplot(1, 1, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bic[i * len(n_components_range):
            (i + 1) * len(n_components_range)],
                                width=.2, color=color))
        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('BIC score per model')
        xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
               .2 * np.floor(bic.argmin() / len(n_components_range))
        plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], cv_types)
        plt.savefig(os.path.join(self.model_store,'GMM_plot.png'))

        self.perform_gmm_training(n_comp=best_n_components,covariance_type=best_cv_type)

        return True

    def load_gmm_model(self):

        if (os.path.exists(os.path.join(self.model_store, 'gmm.pkl'))):
            self.gm = so.load_obj(os.path.join(self.model_store, 'gmm.pkl'))
        else:
            print "CANNOT FIND GMM.PKL in MODEL STORE"
            raise AssertionError

    def get_assigns(self, means, feats):

        dist_matrix = cdist(feats, means)
        assigns = np.argmin(dist_matrix, axis=1)
        return assigns

    def get_centroid_distances(self, means, feats):

        dist_matrix = cdist(feats, means)
        centroid_distances = np.min(dist_matrix, axis=1)

        return centroid_distances

    def update_means(self, feats, cluster_assigns):

        ck = np.zeros((self.means.shape[0]))
        for idx, i in enumerate(feats):
            ck[cluster_assigns[idx]] += 1
            cki = ck[cluster_assigns[idx]]
            self.means[cluster_assigns[idx]] -= (1 / cki) * (
            self.means[cluster_assigns[idx]] - feats[idx]) * self.clustering_lr

    def create_recons(self, n_recons):

        print "CREATING RECONSTRUCTIONS"
        self.set_x_train(np.random.randint(0,len(self.dat_h5)))

        folder_name = 'cuboid_reconstructions'

        if(not os.path.exists(os.path.join(self.model_store,folder_name))):
            os.mkdir(os.path.join(self.model_store,folder_name))


        for i in range(0, n_recons):
                self.do_gif_recon(self.x_train[np.random.randint(0, len(self.x_train))], os.path.join(folder_name,'recon_' + str(i)))

        return True

    def do_gif_recon(self, input_cuboid, name):

        input_cuboid = np.expand_dims(input_cuboid, 0)
        output_cuboid = self.ae.predict(input_cuboid)

        input_cuboid = input_cuboid[0]
        output_cuboid = output_cuboid[0]

        if (self.reverse):
            output_cuboid = np.flip(output_cuboid, axis=2)

        for i in range(0, input_cuboid.shape[-1]/self.n_channels):

            f, (ax1, ax2) = plt.subplots(2, 1)

            if (self.gs):
                ax1.imshow(np.uint8(input_cuboid[:,:,i].reshape(self.size_y, self.size_x)*255.0),cmap='gist_gray')
            else:
                ax1.imshow(np.uint8(input_cuboid[:,:,i*self.n_channels:(i+1)*self.n_channels]*255.0),cmap='gist_gray')

            ax1.set_title('Input Cuboid')
            ax1.set_axis_off()

            if (self.gs):
                ax2.imshow(np.uint8(output_cuboid[:,:,i].reshape(self.size_y, self.size_x)*255.0),cmap='gist_gray')
            else:
                ax2.imshow(np.uint8(output_cuboid[:,:,i*self.n_channels:(i+1)*self.n_channels]*255.0),cmap='gist_gray')
            ax2.set_title('Output Cuboid')
            ax2.set_axis_off()

            plt.axis('off')
            plt.savefig(os.path.join(self.model_store, str(i) + '.png'), bbox_inches='tight')
            plt.close()

        images = []
        for i in range(0, input_cuboid.shape[-1]/self.n_channels):
            images.append(imageio.imread(os.path.join(self.model_store, str(i) + '.png')))

        imageio.mimsave(os.path.join(self.model_store, name + '.gif'), images)

        return True

    def save_gifs(self, input_cuboid, name,custom_msg=None,input=False):

        if(self.reverse and not input):
            input_cuboid = np.flip(input_cuboid,axis=2)

        for i in range(0, input_cuboid.shape[-1]/self.n_channels):

            f, (ax1) = plt.subplots(1, 1)

            if (self.gs):
                ax1.imshow(np.uint8(input_cuboid[:,:,i].reshape(self.size_y, self.size_x)*255.0),cmap='gist_gray')
            else:
                ax1.imshow(np.uint8(input_cuboid[:,:,i*self.n_channels:(i+1)*self.n_channels]*255.0),cmap='gist_gray')

            if(custom_msg):
                ax1.set_title(custom_msg)
            else:
                ax1.set_title('Input Cuboid')

            ax1.set_axis_off()

            plt.axis('off')
            plt.savefig(os.path.join(self.model_store, str(i) + '.png'), bbox_inches='tight')
            plt.close()

        images = []
        for i in range(0, input_cuboid.shape[-1]/self.n_channels):
            images.append(imageio.imread(os.path.join(self.model_store, str(i) + '.png')))

        imageio.mimsave(os.path.join(self.model_store, name + '.gif'), images)

        return True

    def save_weights(self):

        if(self.multi_gpu_model):
            self.ae_single.save_weights(filepath=os.path.join(self.model_store, 'ae_weights.h5'))
        else:
            self.ae.save_weights(filepath=os.path.join(self.model_store, 'ae_weights.h5'))

        self.decoder.save_weights(filepath=os.path.join(self.model_store, 'decoder_weights.h5'))
        self.encoder.save_weights(filepath=os.path.join(self.model_store, 'encoder_weights.h5'))

    def generate_loss_graph(self, graph_name):

        hfm, = plt.plot(self.loss_list, 'ro',label='loss')
        plt.plot(self.loss_list,'b:')
        plt.legend(handles=[hfm])
        plt.title('Losses per training iteration')
        plt.xlabel('Training iteration index')
        plt.ylabel('Loss value (Value of cost function)')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

        return True

    def generate_loss_graph_with_anomaly_gt(self,graph_name):

        with open(os.path.join(self.dat_folder, 'chapter_id_anomaly_list.pkl'), 'rb') as f:
            chapter_id_anomaly_list = pickle.load(f)

        chapter_id_anom_list_completed = []

        for idx, _ in enumerate(self.loss_list):
            chapter_id_anom_list_completed.append(chapter_id_anomaly_list[idx % len(chapter_id_anomaly_list)])

        loss_of_chapters_with_anomaly = np.array(chapter_id_anom_list_completed) * np.array(self.loss_list)

        hfm, = plt.plot(self.loss_list, 'ro',label='loss')
        plt.plot(self.loss_list,'b:')
        hfm2, = plt.plot(loss_of_chapters_with_anomaly, 'go',label='anomalies-present')
        plt.legend(handles=[hfm,hfm2])
        plt.title('Losses per training iteration')
        plt.xlabel('Training iteration index')
        plt.ylabel('Loss value (Value of cost function)')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

        return True

    def get_encodings_and_assigns(self, n_chapters, total_chaps_trained_on):

        encodings_full = []

        for i in range(0, n_chapters):
            idx = np.random.randint(0, total_chaps_trained_on)
            self.set_x_train(idx)
            train_encodings = self.encoder.predict(self.x_train)

            encodings_full.append(train_encodings.tolist())
            del train_encodings

        flat_list_encodings = [item for sublist in encodings_full for item in sublist]

        cluster_assigns = self.get_assigns(self.means, np.array(flat_list_encodings))

        return np.array(flat_list_encodings), cluster_assigns

    def save_gifs_per_cluster_ids(self, n_samples_per_id,total_chaps_trained_on,max_try=100):

        self.features_h5 = h5py.File(os.path.join(self.model_store, 'features.h5'), 'r')

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "START SAVE_GIFS_PER_CLUSTER_IDS"
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

        if (not os.path.exists(os.path.join(self.model_store,'cluster_gifs'))):
            os.mkdir(os.path.join(self.model_store,'cluster_gifs'))

        for i in range (0,self.n_clusters):
            if (not os.path.exists(os.path.join(self.model_store,'cluster_gifs',str(i)))):
                os.mkdir(os.path.join(self.model_store, 'cluster_gifs',str(i)))

        n_saved_each = np.zeros(self.n_clusters)
        n_try = 0

        while(not np.all(n_saved_each >= n_samples_per_id) and n_try < max_try):

            print "SAVE_PROGRESS:", n_saved_each

            idx = np.random.randint(0, total_chaps_trained_on)
            train_encodings = self.set_feats(idx)
            self.set_x_train(idx)
            assigns = self.get_assigns(self.means, np.array(train_encodings))
            distances = self.get_centroid_distances(self.means,np.array(train_encodings))

            for i in range(0,self.n_clusters):
                print "CLUSTER:", i
                if(n_saved_each[i]>=n_samples_per_id):
                    continue

                cuboids_in_cluster_i = self.x_train[assigns==i]
                distances_in_cluster_i = distances[assigns==i]

                cuboids_in_cluster_i,distances_in_cluster_i = shuffle(cuboids_in_cluster_i,distances_in_cluster_i)
                cuboids_in_cluster_i, distances_in_cluster_i = shuffle(cuboids_in_cluster_i, distances_in_cluster_i)

                for idx,j in enumerate(cuboids_in_cluster_i):

                    if(n_saved_each[i]>=n_samples_per_id):
                        break

                    self.save_gifs(j,os.path.join('cluster_gifs',str(i),str(n_saved_each[i])),
                                   custom_msg='distance:'+str(distances_in_cluster_i[idx]),input=True)

                    n_saved_each[i]+=1

            n_try+=1

        return True

    def create_tsne_plot(self, graph_name):

        message_print('DO TSNE FITTING')
        tsne_obj = TSNE(n_components=2, init='pca', random_state=0, verbose=0)

        list_feats = shuffle(self.return_all_encodings())

        select_indexes = np.random.randint(0, len(list_feats), int(0.01 * len(list_feats)))

        train_encodings = list_feats[select_indexes]
        cluster_assigns = self.gm.predict(train_encodings)

        full_array_feats = np.vstack((train_encodings, self.gm.means_))
        n_comps = self.gm.means_.shape[0]
        full_array_labels = np.vstack((cluster_assigns.reshape(len(cluster_assigns), 1), np.ones((n_comps, 1)) * n_comps))

        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        ax = plt.subplot(111)
        plt.scatter(x=train_tsne_embedding[:, 0], y=train_tsne_embedding[:, 1], c=colors,cmap='jet',alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        pickle.dump(ax, file(os.path.join(self.model_store,'tsne2d.pickle'), 'w'))
        plt.close()

    def mean_displacement_distance(self):

        difference = self.means - self.initial_means

        print "####################################"
        print "Mean difference : "
        print  difference
        print "####################################"

        square = np.square(difference)
        sum_squares = np.sum(square, axis=1)

        print "####################################"
        print "Sum squares : "
        print  sum_squares
        print "####################################"

        displacement = np.sqrt(sum_squares)
        print "####################################"
        print "Mean displacement : ", displacement
        print "####################################"

    def generate_mean_displacement_graph(self, graph_name):

        mean_displacement = np.array(self.list_mean_disp)
        print mean_displacement.shape
        list_labels = []
        n = mean_displacement.shape[1]*2

        color = iter(plt.cm.rainbow(np.linspace(0,1,n)))

        for i in range(0, mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:, i],'o', c=next(color), label='centroid ' + str(i + 1))
            plt.plot(mean_displacement[:,i],':',c=next(color))
            list_labels.append(x)


        plt.legend(handles=list_labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Centroid displacements over training iterations')
        plt.xlabel('Training iteration index')
        plt.ylabel('Displacement value')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def generate_assignment_graph(self, graph_name):
        message_print('DO GENERATE ASSIGNMENT GRAPH')
        list_assigns = []
        feats = shuffle(self.return_all_encodings())
        assigns = self.get_assigns(self.means,feats)
        list_assigns.extend(assigns.tolist())

        list_assigns = np.array(list_assigns)

        list_assigns_summed = []

        for i in range(0, self.n_clusters):
            list_assigns_summed.append(np.sum(list_assigns == i))
            print list_assigns_summed[-1]

        plt.bar(x=range(0, self.n_clusters), height=list_assigns_summed, width=0.25)
        plt.title('Cluster Assignments')
        plt.xlabel('Cluster ids')
        plt.ylabel('Number of assignments')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

        return True

    def decode_means(self, graph_name):

        message_print('DO DECODE MEANS')
        means_decoded = self.decoder.predict(self.means)

        folder_name = 'means_decoded'

        if (not os.path.exists(os.path.join(self.model_store, folder_name))):
            os.mkdir(os.path.join(self.model_store, folder_name))

        for i in range(0, len(means_decoded)):
            self.save_gifs(means_decoded[i], os.path.join(folder_name,graph_name + '_' + str(i + 1)))

        return True

    def mean_and_samples(self,n_per_mean):

        message_print('DO MEAN RECONSTRUCTION WITH SAMPLING')

        assert((n_per_mean+1)%3==0),"n_per_mean+1 should be a multiple of 3 for columnwise display"

        folder_name = 'means_and_samples'

        if(not os.path.exists(os.path.join(self.model_store,folder_name))):
            os.mkdir(os.path.join(self.model_store,folder_name))

        points = np.zeros(shape=(self.means.shape[0],n_per_mean+1,self.means.shape[1]))

        for i in range(0,len(self.means)):

            points[i,0,:] = self.means[i]

            for j in range(0,n_per_mean):
                points[i,j+1,:] = self.means[i]+np.random.normal(0.0,0.1,self.means.shape[1])

            self.recon_mean_samples_to_gif(points=points[i],name=os.path.join(folder_name,'mean_and_samples_'+str(i)))

        return True

    def recon_mean_samples_to_gif(self,points,name):
        #Assume that points[0] is the centroid

        decoded = self.decoder.predict(points)

        for i in range(0, decoded.shape[-1]/self.n_channels):
            f, axarr = plt.subplots(len(points) / 3, 3)
            plt.suptitle('Centroid and Nearby Sampling (features) Reconstructed')

            if self.reverse:
                for idx,k in enumerate(decoded):
                    decoded[idx] = np.flip(k, axis=2)

            for j in range(0, len(points) / 3):
                for k in range(0, 3):
                    if (self.gs):
                        axarr[j,k].imshow(np.uint8(decoded[j*3+k,:,:,i].reshape(self.size_y, self.size_x)*255.0),cmap='gist_gray')
                    else:
                        axarr[j,k].imshow(np.uint8(decoded[j*3+k,:,:,i*self.n_channels:(i+1)*self.n_channels]*255.0),cmap='gist_gray')

                    if(j*3+k==0):
                        axarr[j,k].set_title('Mean Cuboid')
                    else:
                        axarr[j,k].set_title('Sampling '+ str(j*3+k+1) + " Cuboid")

                    axarr[j, k].set_axis_off()


            plt.axis('off')
            plt.savefig(os.path.join(self.model_store, str(i) + '.png'), bbox_inches='tight')
            plt.close()

        images = []
        for i in range(0, decoded.shape[-1]/self.n_channels):
            images.append(imageio.imread(os.path.join(self.model_store, str(i) + '.png')))

        imageio.mimsave(os.path.join(self.model_store, name + '.gif'), images)

        return True

    def kmeans_partial_fit_displacement_plot(self):

        with open(os.path.join(self.model_store,'kmeans_fitting.txt')) as f:
            content = f.read().splitlines()

        list_displacements = [float(i.split('=>')[-1]) for i in content]
        list_displacements = list_displacements[10:]

        hfm, = plt.plot(list_displacements, 'ro',label='displacements')
        plt.plot(list_displacements,'b:')
        plt.legend(handles=[hfm])
        plt.title('Sum of mean displacements over kmeans iteration')
        plt.xlabel('Kmeans iteration index')
        plt.ylabel('Displacement value')
        plt.savefig(os.path.join(self.model_store, 'kmeans_fit_plot.png'), bbox_inches='tight')
        plt.close()

        return True

    def kmeans_and_update_pdf(self,list_feats, select_indexes, n_clusters=10, pdf=None):

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30, 40))

        ax.set_xlim([-1, 1])
        ax.set_ylim([0, len(list_feats[select_indexes]) + (n_clusters + 1) * 10])

        clusterer = MiniBatchKMeans(n_clusters=n_clusters, verbose=0, batch_size=256,
                                    compute_labels=False, tol=1e-12, max_no_improvement=200 * 200, random_state=10)

        clusterer.fit(list_feats)

        cluster_labels = clusterer.predict(list_feats[select_indexes])

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(list_feats[select_indexes], cluster_labels)

        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(list_feats[select_indexes], cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]

            y_upper = y_lower + size_cluster_i

            color = cm.get_cmap('Spectral')(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters", fontsize=20)
        ax.set_xlabel("The silhouette coefficient values", fontsize=20)
        ax.set_ylabel("Cluster label", fontsize=20)

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.axvline(x=0.0, color="black", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks(list(np.round(np.linspace(-1, 1, 20), decimals=1)))

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=30)

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        return silhouette_avg

    def perform_num_clusters_analysis(self):

        print "################################"
        print "LOAD_FEATURES"
        print "################################"
        list_feats = shuffle(self.return_all_encodings())
        print "################################"
        print "FEATURES_SHAPE:", list_feats.shape
        print "################################"
        range_n_clusters = range(10,31)
        range_n_clusters.extend((range(35,105,5)))
        print "################################"
        print "RANGE:", range_n_clusters
        print "################################"

        pdf_name = os.path.join(self.model_store, 'silhouette_analysis.pdf')

        select_indexes = np.random.randint(0, len(list_feats), int(0.01 * len(list_feats)))

        if (os.path.exists(pdf_name)):
            os.remove(pdf_name)

        silhouette_avg_score_list = []
        with PdfPages(pdf_name) as pdf:

            for n_clusters in tqdm(range_n_clusters):
                try:
                    avg_score = self.kmeans_and_update_pdf(list_feats,select_indexes,n_clusters, pdf)
                except:
                    avg_score = 0.0

                silhouette_avg_score_list.append(avg_score)

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))

            sns.regplot(x=np.array(range_n_clusters), y=np.array(silhouette_avg_score_list), ax=ax, fit_reg=False,
                        color='r')
            ax.set_xlabel('N_clusters')
            ax.set_ylabel('Average silhouette score')

            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()


        self.n_clusters = range_n_clusters[silhouette_avg_score_list.index(max(silhouette_avg_score_list))]

        print "#########################################################"
        print "CHANGING THE NUMBER OF CLUSTERS TO: ", self.n_clusters
        print "#########################################################"

        so.save_obj(self.n_clusters,os.path.join(self.model_store,'nclusters_optimized.pkl'))

        return True

    def __del__(self):
        print ("Destructor called for SuperAutoEncoder")

class Conv_autoencoder_nostream(Super_autoencoder):

    def __init__(self, model_store, size_y=32, size_x=32, n_channels=3, h_units=256, n_timesteps=5,
                 loss='mse', batch_size=64, n_clusters=10,lr_model=1e-4, lamda=0.01,
                 gs=False,notrain=False,reverse=False,data_folder='data_store',dat_h5=None,
                 large=True, means_tol = 5e-5,means_patience=200,max_fit_tries=500000):

        Super_autoencoder.__init__(self,model_store, size_y=size_y, size_x=size_x, n_channels=n_channels, h_units=h_units, n_timesteps=n_timesteps,
                 loss=loss, batch_size=batch_size, n_clusters=n_clusters,lr_model=lr_model, lamda=lamda,gs=gs, notrain=notrain,
                 reverse=reverse, data_folder=data_folder, dat_h5=dat_h5,large=large, means_tol=means_tol,
                 means_patience=means_patience, max_fit_tries=max_fit_tries)

        if (self.large):
            f1 = 64
            f2 = 128
            f3 = 256
            f4 = 512
            f5 = 256
            f6 = 128
            f7 = 64
            f8 = 32
        else:
            f1 = 16
            f2 = 32
            f3 = 64
            f4 = 256
            f5 = 128
            f6 = 64
            f7 = 32
            f8 = 16

        if(size_x==48):
            resize_factor = 16
            avgpool_4 = True

        else:
            resize_factor = 8
            avgpool_4 = False

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels * n_timesteps)) #24,24 #48,48
        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(f1, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 12x12 #24,24

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(f2, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 6x6 #12,12

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(f3, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)

        if(avgpool_4):
            x1 = AveragePooling2D(pool_size=(4, 4))(x1)  #3x3
        else:
            x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 3x3

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)


        dec1 = Dense(units=(size_y / resize_factor) * (size_x / resize_factor) * f4)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x / resize_factor, size_y / resize_factor, f4))

        if(avgpool_4):
            dec4 = UpSampling2D(size=(4, 4))
        else:
            dec4 = UpSampling2D(size=(2, 2))

        dec5 = Conv2D(f5, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(f6, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(f7, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(f8, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        recon = Conv2D(n_channels * n_timesteps, (3, 3), activation='sigmoid', padding='same')

        ae1 = dec1(encoder)
        ae2 = dec2(ae1)
        ae3 = dec3(ae2)
        ae4 = dec4(ae3)
        ae5 = dec5(ae4)
        ae6 = dec6(ae5)
        ae7 = dec7(ae6)
        ae8 = dec8(ae7)
        ae9 = dec9(ae8)
        ae10 = dec10(ae9)
        ae11 = dec11(ae10)
        ae12 = dec12(ae11)
        ae13 = dec13(ae12)
        ae14 = dec14(ae13)
        ae15 = dec15(ae14)
        ae16 = dec16(ae15)
        ae17 = dec17(ae16)
        ae18 = recon(ae17)

        self.adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if (loss == 'mse'):
            self.loss_fn = mean_squared_error
        elif (loss == 'dssim'):
            self.loss_fn = dssim
        elif (loss == 'bce'):
            self.loss_fn = binary_crossentropy

        self.encoder = Model(inputs=[inp], outputs=[encoder])


        dinp = Input(shape=(h_units,))

        de1 = dec1(dinp)
        de2 = dec2(de1)
        de3 = dec3(de2)
        de4 = dec4(de3)
        de5 = dec5(de4)
        de6 = dec6(de5)
        de7 = dec7(de6)
        de8 = dec8(de7)
        de9 = dec9(de8)
        de10 = dec10(de9)
        de11 = dec11(de10)
        de12 = dec12(de11)
        de13 = dec13(de12)
        de14 = dec14(de13)
        de15 = dec15(de14)
        de16 = dec16(de15)
        de17 = dec17(de16)
        de18 = recon(de17)


        self.y_obj = CustomClusterLayer(self.loss_fn, self.means, self.lamda, self.n_clusters, h_units,reverse=reverse)

        y = self.y_obj([inp, ae18, encoder])

        self.decoder = Model(inputs=[dinp], outputs=[de18])

        self.ae = Model(inputs=[inp], outputs=[y])

        self.ae.compile(optimizer=self.adam_ae, loss=None)

        if (os.path.isfile(os.path.join(model_store, 'ae_weights.h5'))):
            print "LOADING AE MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store, 'ae_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'decoder_weights.h5'))):
            print "LOADING DECODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.decoder.load_weights(os.path.join(model_store, 'decoder_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'encoder_weights.h5'))):
            print "LOADING ENCODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.encoder.load_weights(os.path.join(model_store, 'encoder_weights.h5'))

        if (notrain):
            # self.initial_means = np.load(os.path.join(self.model_store, 'initial_means.npy'))

            if (os.path.isfile(os.path.join(model_store, 'means.npy'))):
                self.means = np.load(os.path.join(self.model_store, 'means.npy'))

            if(os.path.isfile(os.path.join(self.model_store, 'losslist.pkl'))):
                with open(os.path.join(self.model_store, 'losslist.pkl'), 'rb') as f:
                    self.loss_list = pickle.load(f)

            if (os.path.isfile(os.path.join(self.model_store, 'meandisp.pkl'))):
                with open(os.path.join(self.model_store, 'meandisp.pkl'), 'rb') as f:
                    self.list_mean_disp = pickle.load(f)

            if (os.path.isfile(os.path.join(self.model_store, 'basis_dict.pkl'))):
                with open(os.path.join(self.model_store, 'basis_dict.pkl'), 'rb') as f:
                    self.dict = pickle.load(f)


        self.ae.summary()

class Conv_autoencoder_nostream_UCSD_noh(Super_autoencoder):


    def __init__(self, model_store, size_y=32, size_x=32, n_channels=3, n_timesteps=5,
                 loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-2, lr_model=1e-4, lamda=0.01,
                 lamda_assign=0.01, gs=False,notrain=False,reverse=False,data_folder='data_store',dat_h5=None,
                 large=True, means_tol = 1e-5,means_patience=200,max_fit_tries=500000):

        if (large):
            f1 = 64
            f2 = 128
            f3 = 256
            f5 = 256
            f6 = 128
            f7 = 64
            f8 = 32
        else:
            f1 = 16
            f2 = 32
            f5 = 128
            f6 = 64
            f7 = 32
            f8 = 16



        self.h_units = (f3/8)*(size_x/(2**4))*(size_x/(2**4))

        Super_autoencoder.__init__(self, model_store, size_y=size_y, size_x=size_x, n_channels=n_channels,
                                   h_units=self.h_units, n_timesteps=n_timesteps,
                                   loss=loss, batch_size=batch_size, n_clusters=n_clusters,
                                   lr_model=lr_model, lamda=lamda, gs=gs, notrain=notrain, reverse=reverse,
                                   data_folder=data_folder,
                                   dat_h5=dat_h5, large=large, means_tol=means_tol, means_patience=means_patience,
                                   max_fit_tries=max_fit_tries)

        self.means = np.zeros((n_clusters, self.h_units))

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels * n_timesteps))
        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(f1, (4, 4), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 24x24

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(f2, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 12x12

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(f3, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 6x6

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(f3/8, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 3x3

        encoder = Flatten()(x1)

        dec3 = Reshape((size_x / 16, size_y / 16, f3/8)) #3x3

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(f5, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 6x6

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(f5, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  #12x12

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(f6, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  #24x24

        dec16 = UpSampling2D(size=(2, 2))
        dec17 = Conv2D(f7, (4, 4), padding='same')
        dec18 = LeakyReLU(alpha=0.2)
        dec19 = BatchNormalization()  #48x48

        dec20 = Conv2D(f8, (3, 3), padding='same')
        dec21 = LeakyReLU(alpha=0.2)
        recon = Conv2D(n_channels * n_timesteps, (3, 3), activation='sigmoid', padding='same') #48x48

        ae3 = dec3(encoder)
        ae4 = dec4(ae3)
        ae5 = dec5(ae4)
        ae6 = dec6(ae5)
        ae7 = dec7(ae6)
        ae8 = dec8(ae7)
        ae9 = dec9(ae8)
        ae10 = dec10(ae9)
        ae11 = dec11(ae10)
        ae12 = dec12(ae11)
        ae13 = dec13(ae12)
        ae14 = dec14(ae13)
        ae15 = dec15(ae14)
        ae16 = dec16(ae15)
        ae17 = dec17(ae16)
        ae18 = dec18(ae17)
        ae19 = dec19(ae18)
        ae20 = dec20(ae19)
        ae21 = dec21(ae20)
        ae22 = recon(ae21)

        self.adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=8)

        if (loss == 'mse'):
            self.loss_fn = mean_squared_error
        elif (loss == 'dssim'):
            self.loss_fn = dssim
        elif (loss == 'bce'):
            self.loss_fn = binary_crossentropy

        self.encoder = Model(inputs=[inp], outputs=[encoder])

        dinp = Input(shape=(self.h_units,))


        de3 = dec3(dinp)
        de4 = dec4(de3)
        de5 = dec5(de4)
        de6 = dec6(de5)
        de7 = dec7(de6)
        de8 = dec8(de7)
        de9 = dec9(de8)
        de10 = dec10(de9)
        de11 = dec11(de10)
        de12 = dec12(de11)
        de13 = dec13(de12)
        de14 = dec14(de13)
        de15 = dec15(de14)
        de16 = dec16(de15)
        de17 = dec17(de16)
        de18 = dec18(de17)
        de19 = dec19(de18)
        de20 = dec20(de19)
        de21 = dec21(de20)
        de22 = recon(de21)


        self.y_obj = CustomClusterLayer(self.loss_fn, self.means, self.lamda, self.n_clusters, self.h_units,reverse=reverse)

        y = self.y_obj([inp, ae22, encoder])

        self.decoder = Model(inputs=[dinp], outputs=[de22])

        self.ae = Model(inputs=[inp], outputs=[y])

        self.ae.compile(optimizer=self.adam_ae, loss=None)


        if (os.path.isfile(os.path.join(model_store, 'ae_weights.h5'))):
            print "LOADING AE MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store, 'ae_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'decoder_weights.h5'))):
            print "LOADING DECODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.decoder.load_weights(os.path.join(model_store, 'decoder_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'encoder_weights.h5'))):
            print "LOADING ENCODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.encoder.load_weights(os.path.join(model_store, 'encoder_weights.h5'))

        if (notrain):
            # self.initial_means = np.load(os.path.join(self.model_store, 'initial_means.npy'))
            self.means = np.load(os.path.join(self.model_store, 'means.npy'))

            if (os.path.isfile(os.path.join(self.model_store, 'losslist.pkl'))):
                with open(os.path.join(self.model_store, 'losslist.pkl'), 'rb') as f:
                    self.loss_list = pickle.load(f)

            if (os.path.isfile(os.path.join(self.model_store, 'meandisp.pkl'))):
                with open(os.path.join(self.model_store, 'meandisp.pkl'), 'rb') as f:
                    self.list_mean_disp = pickle.load(f)

            if (os.path.isfile(os.path.join(self.model_store, 'basis_dict.pkl'))):
                with open(os.path.join(self.model_store, 'basis_dict.pkl'), 'rb') as f:
                    self.dict = pickle.load(f)

        self.ae.summary()

class Conv_autoencoder_nostream_UCSD_h(Super_autoencoder):


    def __init__(self, model_store, size_y=32, size_x=32, n_channels=3, h_units=256, n_timesteps=5,
                 loss='mse', batch_size=64, n_clusters=10, lr_model=1e-4, lamda=0.01,
                 gs=False,notrain=False,reverse=False,data_folder='data_store',dat_h5=None,
                 large=True, means_tol = 1e-5,means_patience=200,max_fit_tries=500000):

        Super_autoencoder.__init__(self, model_store, size_y=size_y, size_x=size_x, n_channels=n_channels,
                                   h_units=h_units, n_timesteps=n_timesteps,
                                   loss=loss, batch_size=batch_size, n_clusters=n_clusters,
                                   lr_model=lr_model, lamda=lamda, gs=gs, notrain=notrain, reverse=reverse,
                                   data_folder=data_folder,
                                   dat_h5=dat_h5, large=large, means_tol=means_tol, means_patience=means_patience,
                                   max_fit_tries=max_fit_tries)

        if (self.large):
            f1 = 64
            f2 = 128
            f3 = 256
            f4 = 512
            f5 = 256
            f6 = 128
            f7 = 64
            f8 = 32
        else:
            f1 = 16
            f2 = 32
            f3 = 64
            f4 = 256
            f5 = 128
            f6 = 64
            f7 = 32
            f8 = 16

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels * n_timesteps))
        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(f1, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 24x24

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(f2, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 12x12

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(f3, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 6x6

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(f3, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.3)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = AveragePooling2D(pool_size=(2, 2))(x1)  # 3x3

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)

        dec1 = Dense(units=(size_y / 16) * (size_x / 16) * f4)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x / 16, size_y / 16, f4)) #3x3

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(f5, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 6x6

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(f5, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  #12x12

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(f6, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  #24x24

        dec16 = UpSampling2D(size=(2, 2))
        dec17 = Conv2D(f7, (3, 3), padding='same')
        dec18 = LeakyReLU(alpha=0.2)
        dec19 = BatchNormalization()  #48x48

        dec20 = Conv2D(f8, (3, 3), padding='same')
        dec21 = LeakyReLU(alpha=0.2)
        recon = Conv2D(n_channels * n_timesteps, (3, 3), activation='sigmoid', padding='same') #48x48

        ae1 = dec1(encoder)
        ae2 = dec2(ae1)
        ae3 = dec3(ae2)
        ae4 = dec4(ae3)
        ae5 = dec5(ae4)
        ae6 = dec6(ae5)
        ae7 = dec7(ae6)
        ae8 = dec8(ae7)
        ae9 = dec9(ae8)
        ae10 = dec10(ae9)
        ae11 = dec11(ae10)
        ae12 = dec12(ae11)
        ae13 = dec13(ae12)
        ae14 = dec14(ae13)
        ae15 = dec15(ae14)
        ae16 = dec16(ae15)
        ae17 = dec17(ae16)
        ae18 = dec18(ae17)
        ae19 = dec19(ae18)
        ae20 = dec20(ae19)
        ae21 = dec21(ae20)
        ae22 = recon(ae21)

        self.adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=8)

        if (loss == 'mse'):
            self.loss_fn = mean_squared_error
        elif (loss == 'dssim'):
            self.loss_fn = dssim
        elif (loss == 'bce'):
            self.loss_fn = binary_crossentropy

        self.encoder = Model(inputs=[inp], outputs=[encoder])

        dinp = Input(shape=(h_units,))

        de1 = dec1(dinp)
        de2 = dec2(de1)
        de3 = dec3(de2)
        de4 = dec4(de3)
        de5 = dec5(de4)
        de6 = dec6(de5)
        de7 = dec7(de6)
        de8 = dec8(de7)
        de9 = dec9(de8)
        de10 = dec10(de9)
        de11 = dec11(de10)
        de12 = dec12(de11)
        de13 = dec13(de12)
        de14 = dec14(de13)
        de15 = dec15(de14)
        de16 = dec16(de15)
        de17 = dec17(de16)
        de18 = dec18(de17)
        de19 = dec19(de18)
        de20 = dec20(de19)
        de21 = dec21(de20)
        de22 = recon(de21)


        self.y_obj = CustomClusterLayer(self.loss_fn, self.means, self.lamda, self.n_clusters, h_units,reverse=reverse)

        y = self.y_obj([inp, ae22, encoder])

        self.decoder = Model(inputs=[dinp], outputs=[de22])

        self.ae = Model(inputs=[inp], outputs=[y])

        self.ae.compile(optimizer=self.adam_ae, loss=None)

        if (os.path.isfile(os.path.join(model_store, 'ae_weights.h5'))):
            print "LOADING AE MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store, 'ae_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'decoder_weights.h5'))):
            print "LOADING DECODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.decoder.load_weights(os.path.join(model_store, 'decoder_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'encoder_weights.h5'))):
            print "LOADING ENCODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.encoder.load_weights(os.path.join(model_store, 'encoder_weights.h5'))

        if (notrain):
            # self.initial_means = np.load(os.path.join(self.model_store, 'initial_means.npy'))
            self.means = np.load(os.path.join(self.model_store, 'means.npy'))

            if (os.path.isfile(os.path.join(self.model_store, 'losslist.pkl'))):
                with open(os.path.join(self.model_store, 'losslist.pkl'), 'rb') as f:
                    self.loss_list = pickle.load(f)

            if (os.path.isfile(os.path.join(self.model_store, 'meandisp.pkl'))):
                with open(os.path.join(self.model_store, 'meandisp.pkl'), 'rb') as f:
                    self.list_mean_disp = pickle.load(f)

            if (os.path.isfile(os.path.join(self.model_store, 'basis_dict.pkl'))):
                with open(os.path.join(self.model_store, 'basis_dict.pkl'), 'rb') as f:
                    self.dict = pickle.load(f)

        self.ae.summary()
