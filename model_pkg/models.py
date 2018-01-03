import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.cluster import KMeans,MiniBatchKMeans
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Lambda, Reshape, GaussianNoise, Conv2D, SpatialDropout2D, LeakyReLU
from keras.layers import Dropout, MaxPooling2D, Layer, merge
from keras.layers import UpSampling2D, Flatten, BatchNormalization,ConvLSTM2D,TimeDistributed
from keras.layers.core import Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.losses import mean_squared_error,binary_crossentropy
import os
from keras_contrib.losses import DSSIMObjective
from keras.datasets import cifar10,mnist,fashion_mnist
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
import keras.backend as K
from mpl_toolkits.mplot3d import Axes3D
import imageio
import tensorflow as tf

def small_2d_conv_net(size_y=32,size_x=32,n_channels=1,n_frames=5,h_units=100):

    inp = Input(shape=(size_y,size_x,n_channels,n_frames))

    x1 = Reshape(target_shape=(size_y,size_x,n_channels*n_frames))(inp)
    x1 = GaussianNoise(0.05)(x1)
    x1 = Conv2D(64,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.5)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1)

    x1 = GaussianNoise(0.03)(x1)
    x1 = Conv2D(128,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.5)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1)

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(128,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.5)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = BatchNormalization()(x1)

    # x1 = GaussianNoise(0.01)(x1)
    # x1 = Conv2D(256,(3,3),padding='same')(x1)
    # x1 = SpatialDropout2D(0.3)(x1)
    # x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = MaxPooling2D(pool_size=(2,2))(x1)

    x1 = Flatten()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(units=h_units)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)

    x1 = Dense(units=(size_y/4)*(size_x/4)*128)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = Reshape((size_x/4,size_y/4,128))(x1)

    # x1 = UpSampling2D(size=(2,2))(x1)
    # x1 = Conv2D(256,(3,3),padding='same')(x1)
    # x1 = LeakyReLU(alpha=0.2)(x1)

    x1 = UpSampling2D(size=(2,2))(x1)
    x1 = Conv2D(128,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    # x1 = BatchNormalization()(x1)

    x1 = UpSampling2D(size=(2,2))(x1)
    x1 = Conv2D(64,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)



    x1 = Conv2D(n_frames,(3,3),activation='tanh', padding='same')(x1)
    x1 = Reshape(target_shape=(size_y,size_x,n_channels,n_frames))(x1)

    model = Model(inputs=[inp], outputs=[x1])
    rmsprop = RMSprop(lr=0.0001)

    model.compile(optimizer=rmsprop,loss='mse')

    return model

def small_2d_conv_net_cifar(size_y=32,size_x=32,n_channels=3,h_units=10):

    inp = Input(shape=(size_y,size_x,n_channels))
    x1 = GaussianNoise(0.05)(inp)
    x1 = Conv2D(64,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.5)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1) #16x16

    x1 = GaussianNoise(0.03)(x1)
    x1 = Conv2D(128,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.5)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2,2))(x1) #8x8

    x1 = GaussianNoise(0.02)(x1)
    x1 = Conv2D(128,(3,3),padding='same')(x1)
    x1 = SpatialDropout2D(0.5)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1) #4x4

    x1 = Flatten()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(units=h_units)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)

    x1 = Dense(units=(size_y/8)*(size_x/8)*128)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = Reshape((size_x/8,size_y/8,128))(x1)

    x1 = UpSampling2D(size=(2,2))(x1)
    x1 = Conv2D(256,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1) #8x8

    x1 = UpSampling2D(size=(2,2))(x1)
    x1 = Conv2D(128,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1) #16x16

    x1 = UpSampling2D(size=(2,2))(x1)
    x1 = Conv2D(64,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = BatchNormalization()(x1)#32x32

    x1 = Conv2D(32,(3,3),padding='same')(x1)
    x1 = LeakyReLU(alpha=0.2)(x1) #32x32

    x1 = Conv2D(n_channels,(3,3),activation='tanh', padding='same')(x1)


    model = Model(inputs=[inp], outputs=[x1])
    rmsprop = RMSprop(lr=1e-3)

    dssim = DSSIMObjective(kernel_size=4)

    model.compile(optimizer=rmsprop,loss=dssim)

    return model

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)

class CustomClusterLayer(Layer):

    def __init__(self,loss_fn,means,lamda,n_clusters,hid,**kwargs):

        self.is_placeholder = True
        self.loss_fn=loss_fn

        self.means=K.variable(value=K.cast_to_floatx(means),name='means_tensor_in')
        self.lamda=K.variable(value=K.cast_to_floatx(lamda),name='lamda_tensor_in')

        self.n_clusters = n_clusters
        self.hid = hid
        self.cl_loss_wt = K.variable(value=K.cast_to_floatx(x=0.0),name='cl_loss_wt_in_model')

        super(CustomClusterLayer, self).__init__(**kwargs)

    def custom_loss_clustering(self,x_true, x_pred, encoded_feats):

        loss = K.mean(self.loss_fn(x_true, x_pred))
        centroids = self.means

        M = self.n_clusters
        N = K.shape(x_true)[0]

        rep_centroids = K.reshape(K.tile(centroids, [N, 1]), [N, M, self.hid])
        rep_points = K.reshape(K.tile(encoded_feats, [1, M]), [N, M, self.hid])
        sum_squares = K.tf.reduce_sum(K.square(rep_points - rep_centroids),reduction_indices=2)
        best_centroids = K.argmin(sum_squares, 1)

        cl_loss = self.cl_loss_wt * (self.lamda / 2.0) * K.mean(K.sqrt(K.sum(K.square(encoded_feats-K.gather(centroids,best_centroids)),axis=1)))

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
            loss = K.mean(self.loss_fn(K.reverse(x_true,axes=1), x_pred))
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
        sum_all_unequal = K.mean(K.cast(unequal_assignments,dtype='float32'))
        assignment_loss = self.cl_loss_wt * (self.assignment_lamda) * sum_all_unequal

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

class CIFAR_model_vanilla:


    def __init__(self,model_store,size_y=32,size_x=32,n_channels=3,h_units=64,subtract_pixel_mean=False,loss='mse'):

        self.model_store = model_store
        self.loss = loss

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        #Get CIFAR 10 data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        self.y_train = self.y_train.reshape(self.y_train.shape[0],1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0],1)

        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        self.x_train_mean = np.mean(self.x_train, axis=0)

        if subtract_pixel_mean:
            self.subtract_pixel_mean = 1
            self.x_train -= self.x_train_mean
            self.x_test -= self.x_train_mean
        else:
            self.subtract_pixel_mean = 0

        self.x_train,self.y_train = shuffle(self.x_train,self.y_train)
        self.x_test,self.y_test = shuffle(self.x_test,self.y_test)

        #Change Data format to channels_last
        img_rows = self.x_train.shape[1]
        img_cols = self.x_train.shape[2]
        channels = self.x_train.shape[3]

        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, channels)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, channels)

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels))

        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(64, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 16x16

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(128, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 8x8

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(256, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = BatchNormalization()(x1)
        encoder = LeakyReLU(alpha=0.2)(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 512 )
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x / 8, size_y / 8, 512))

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(256, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(128, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(64, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(32, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        if(self.subtract_pixel_mean):
            recon = Conv2D(n_channels, (3, 3), activation='tanh', padding='same')
        else:
            recon = Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')

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


        self.ae = Model(inputs=[inp], outputs=[ae18])
        rmsprop = RMSprop(lr=1e-3)
        dssim = DSSIMObjective(kernel_size=4)

        if(loss=='mse'):
            self.ae.compile(optimizer=rmsprop, loss='mse')
        elif(loss=='dssim'):
            self.ae.compile(optimizer=rmsprop, loss=dssim)
        elif(loss=='bce'):
            self.ae.compile(optimizer=rmsprop, loss='binary_crossentropy')

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

        self.decoder = Model(inputs=[dinp], outputs=[de18])

        if(os.path.isfile(os.path.join(model_store,'weights.h5'))):
            print "LOADING MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store,'weights.h5'))


    def fit_model_ae(self,verbose):

        es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        mcp = ModelCheckpoint(filepath=os.path.join(self.model_store,'weights.h5'), monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True)
        tb = TensorBoard(os.path.join(self.model_store,'logs'))
        self.ae.fit(self.x_train, self.x_train, shuffle=True, epochs=200, callbacks=[es, rlr, mcp,tb],validation_split=0.2,verbose=verbose, batch_size=64)


    def create_recon_images(self,num_recons):

        for l in range(0,num_recons):
            image_index = np.random.randint(0, len(self.x_test))
            xtm = self.x_train_mean*self.subtract_pixel_mean
            image_to_be_recon = np.uint8((self.x_test[image_index]+xtm)*255)
            image_to_be_recon_shaped = np.expand_dims(self.x_test[image_index], axis=0)
            class_of_image = self.y_test[image_index]

            reconstruction = np.uint8((self.ae.predict(image_to_be_recon_shaped)[0]+xtm)*255)
            encoding = self.encoder.predict(image_to_be_recon_shaped)[0]

            # print "########################"
            # print "ENCODING: ", encoding
            # print "########################"

            n = encoding.shape[0]
            samples = np.ones((7,n))*encoding + np.random.normal(0.0,0.1,size=(7,n))

            # print "########################"
            # print "SAMPLES: ", samples
            # print "########################"

            predictions_samples = np.uint8((self.decoder.predict(samples)+xtm)*255)

            f, axarr = plt.subplots(3, 3)
            plt.suptitle('Recon-class :' + str(class_of_image))
            axarr[0, 0].imshow(image_to_be_recon)
            axarr[0, 0].set_title('Original Image')

            axarr[0, 1].imshow(reconstruction)
            axarr[0, 1].set_title('R-Image')

            for i in range(0, 3):
                for j in range(0,3):

                    if((i==0 and j==0) or (i==0 and j==1)):
                        continue

                    axarr[i,j].imshow(predictions_samples[i*3+j-2])
                    axarr[i,j].set_title('R-Sampling :' + str(i*3+j-2))


            plt.savefig(os.path.join(self.model_store, 'recon-set' + str(l)))
            plt.close()


    def create_tsne_plot(self,graph_name):

        tsne_obj = TSNE(n_components=2, init='pca', random_state=0, verbose=1)
        train_encodings = self.encoder.predict(self.x_train)
        test_encodings = self.encoder.predict(self.x_test)
        full_array_feats = np.vstack((train_encodings, test_encodings))
        full_array_labels = np.vstack((self.y_train, self.y_test))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        plt.scatter(x=train_tsne_embedding[:, 0], y=train_tsne_embedding[:, 1], c=colors, cmap=plt.get_cmap('Set3'))
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

class CIFAR_model_cluster:

    loss_list_fm=[]
    loss_list_dec = []
    means = None
    initial_means = None
    list_mean_disp = []

    def __init__(self,model_store,size_y=32,size_x=32,n_channels=3,h_units=64,subtract_pixel_mean=False,loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-5,lr_model=1e-4):

        self.clustering_lr = clustering_lr
        self.batch_size=batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        #Get CIFAR 10 data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        self.y_train = self.y_train.reshape(self.y_train.shape[0],1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0],1)

        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        self.x_train_mean = np.mean(self.x_train, axis=0)

        if subtract_pixel_mean:
            self.subtract_pixel_mean = 1
            self.x_train -= self.x_train_mean
            self.x_test -= self.x_train_mean
        else:
            self.subtract_pixel_mean = 0

        self.x_train,self.y_train = shuffle(self.x_train,self.y_train)
        self.x_test,self.y_test = shuffle(self.x_test,self.y_test)

        #Change Data format to channels_last
        img_rows = self.x_train.shape[1]
        img_cols = self.x_train.shape[2]
        channels = self.x_train.shape[3]

        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, channels)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, channels)

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels))

        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(64, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 16x16

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(128, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 8x8

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(256, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)


        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 512 )
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x / 8, size_y / 8, 512))

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(256, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(128, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(64, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(32, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        if(self.subtract_pixel_mean):
            recon = Conv2D(n_channels, (3, 3), activation='tanh', padding='same')
        else:
            recon = Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')

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


        self.ae = Model(inputs=[inp], outputs=[ae18])
        adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if(loss=='mse'):
            self.loss_compile = 'mse'
        elif(loss=='dssim'):
            self.loss_compile = dssim
        elif(loss=='bce'):
            self.loss_compile = 'binary_crossentropy'

        self.ae.compile(optimizer=adam_ae, loss=self.loss_compile)


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

        self.decoder = Model(inputs=[dinp], outputs=[de18])
        adam_dec = Adam(lr=self.lr_model)
        self.decoder.compile(optimizer=adam_dec,loss=self.loss_compile)

        if(os.path.isfile(os.path.join(model_store,'weights.h5'))):
            print "LOADING MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store,'weights.h5'))

    def make_trainable(self,net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def fit_model_ae(self,verbose=1,n_initial_epochs=10,n_train_iterations=1e5):

        #start_two_epoch_of_training
        self.ae.fit(self.x_train, self.x_train, shuffle=True, epochs=n_initial_epochs, batch_size=64, verbose=verbose)

        predicted_feats = self.encoder.predict(self.x_train)

        #Get means of predicted features
        km = KMeans(n_clusters=self.n_clusters, verbose=0, n_jobs=-1)
        km.fit(predicted_feats)

        self.means = np.copy(km.cluster_centers_)
        self.initial_means = np.copy(km.cluster_centers_)

        del km


        for i in tqdm(range(0, int(n_train_iterations))):

            #Get batch
            batch_idx=np.random.randint(0,len(self.x_train),size=self.batch_size)
            batch = self.x_train[batch_idx]

            #Get cluster-feats
            encoded_feats = self.encoder.predict(batch)
            cluster_assigns = self.get_assigns(self.means,encoded_feats)

            means_pre = np.copy(self.means)

            #Update means
            for j in range(0,self.n_clusters):
                feats_belonging_to_cluster = encoded_feats[cluster_assigns==j]
                if(feats_belonging_to_cluster.shape[0]>0):
                    mean_fbc = np.mean(feats_belonging_to_cluster,axis=0)
                    lr = self.clustering_lr/(i+1)
                    self.means[j]=(1-lr)*self.means[j]+lr*mean_fbc



            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))
            #Get features to be sent to decoder for learning
            decoder_feats = self.means[cluster_assigns] + np.random.normal(loc=0.0,scale=np.var(np.var(encoded_feats,axis=0)),size=encoded_feats.shape)

            #Train decoder
            self.loss_list_dec.append(self.decoder.train_on_batch(decoder_feats, batch))
            #Fix_decoder
            self.make_trainable(self.decoder,False)

            #Train Full model
            self.loss_list_fm.append(self.ae.train_on_batch(batch,batch))
            if(np.random.rand()<0.3):
                n_batch_idx = np.random.randint(0, len(self.x_train), size=self.batch_size)
                n_batch = self.x_train[n_batch_idx]
                self.loss_list_fm.append(self.ae.train_on_batch(n_batch, n_batch))

            #Release decoder
            self.make_trainable(self.decoder,True)


        self.ae.save_weights(filepath=os.path.join(self.model_store,'weights.h5'))

    def get_assigns(self,means,feats):

        dist_matrix = cdist(feats,means)
        assigns = np.argmin(dist_matrix,axis=1)
        return assigns

    def create_recon_images(self,num_recons):

        for l in range(0,num_recons):
            image_index = np.random.randint(0, len(self.x_test))
            xtm = self.x_train_mean*self.subtract_pixel_mean
            image_to_be_recon = np.uint8((self.x_test[image_index]+xtm)*255)
            image_to_be_recon_shaped = np.expand_dims(self.x_test[image_index], axis=0)
            class_of_image = self.y_test[image_index]

            reconstruction = np.uint8((self.ae.predict(image_to_be_recon_shaped)[0]+xtm)*255)
            encoding = self.encoder.predict(image_to_be_recon_shaped)[0]

            # print "########################"
            # print "ENCODING: ", encoding
            # print "########################"

            n = encoding.shape[0]
            samples = np.ones((7,n))*encoding + np.random.normal(0.0,0.1,size=(7,n))

            # print "########################"
            # print "SAMPLES: ", samples
            # print "########################"

            predictions_samples = np.uint8((self.decoder.predict(samples)+xtm)*255)

            f, axarr = plt.subplots(3, 3)
            plt.suptitle('Recon-class :' + str(class_of_image))
            axarr[0, 0].imshow(image_to_be_recon)
            axarr[0, 0].set_title('Original Image')

            axarr[0, 1].imshow(reconstruction)
            axarr[0, 1].set_title('R-Image')

            for i in range(0, 3):
                for j in range(0,3):

                    if((i==0 and j==0) or (i==0 and j==1)):
                        continue

                    axarr[i,j].imshow(predictions_samples[i*3+j-2])
                    axarr[i,j].set_title('R-Sampling :' + str(i*3+j-2))


            plt.savefig(os.path.join(self.model_store, 'recon-set' + str(l)))
            plt.close()

    def generate_loss_graph(self,graph_name):

        hfm, = plt.plot(self.loss_list_fm,label='full_model')
        hdec, = plt.plot(self.loss_list_dec,label='decoder')
        plt.legend(handles=[hfm,hdec])
        plt.title('Losses per iteration')
        plt.xlabel('Iteration Index')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot(self,graph_name):

        tsne_obj = TSNE(n_components=2, init='pca', random_state=0, verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        test_encodings = self.encoder.predict(self.x_test)
        means=self.means
        full_array_feats = np.vstack((train_encodings,test_encodings,means))
        full_array_labels = np.vstack((self.y_train,self.y_test,np.ones((self.n_clusters,1))*10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        plt.scatter(x=train_tsne_embedding[:, 0], y=train_tsne_embedding[:, 1], c=colors, cmap=plt.get_cmap('Set3'),alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def mean_displacement_distance(self):
        displacement = np.sqrt(np.sum(np.square(self.means-self.initial_means),axis=1))
        print displacement

    def generate_mean_displacement_graph(self,graph_name):

        mean_displacement = np.array(self.list_mean_disp)
        list_labels=[]

        for i in range(0,mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:,i],label='cluster_'+str(i+1))
            list_labels.append(x)

        plt.legend(handles=list_labels)
        plt.title('Mean displacements over iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Displacement Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

class MNIST_model_vanilla:

    loss_list_fm=[]
    loss_list_dec = []
    means = None
    initial_means = None

    def __init__(self,model_store,size_y=32,size_x=32,n_channels=1,h_units=64,subtract_pixel_mean=False,loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-5):

        self.clustering_lr = clustering_lr
        self.batch_size=batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        #Get MNIST 10 data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.y_train = self.y_train.reshape(self.y_train.shape[0],1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0],1)

        self.x_train = np.pad(self.x_train,pad_width=((0,0),(2,2),(2,2)),mode='constant',constant_values=0)
        self.x_test = np.pad(self.x_test, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)

        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        self.x_train_mean = np.mean(self.x_train, axis=0)

        if subtract_pixel_mean:
            self.subtract_pixel_mean = 1
            self.x_train -= self.x_train_mean
            self.x_test -= self.x_train_mean
        else:
            self.subtract_pixel_mean = 0

        self.x_train,self.y_train = shuffle(self.x_train,self.y_train)
        self.x_test,self.y_test = shuffle(self.x_test,self.y_test)

        #Change Data format to channels_last
        img_rows = self.x_train.shape[1]
        img_cols = self.x_train.shape[2]
        channels = 1

        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, channels)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, channels)

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels))

        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(8, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 14x14

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(16, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 7x7

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(32, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = BatchNormalization()(x1)
        encoder = LeakyReLU(alpha=0.2)(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 64)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x/8 , size_y/8 , 64))

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(32, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(16, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(8, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(8, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        if(self.subtract_pixel_mean):
            recon = Conv2D(n_channels, (3, 3), activation='tanh', padding='same')
        else:
            recon = Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')

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


        self.ae = Model(inputs=[inp], outputs=[ae18])
        adam = Adam(lr=1e-3)
        dssim = DSSIMObjective(kernel_size=4)

        if(loss=='mse'):
            self.loss_compile = 'mse'

        elif(loss=='dssim'):
            self.loss_compile = dssim
        elif(loss=='bce'):
            self.loss_compile = 'binary_crossentropy'

        self.ae.compile(optimizer=adam, loss=self.loss_compile)


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

        self.decoder = Model(inputs=[dinp], outputs=[de18])
        self.decoder.compile(optimizer=adam,loss=self.loss_compile)

        if(os.path.isfile(os.path.join(model_store,'weights.h5'))):
            print "LOADING MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store,'weights.h5'))


    def fit_model_ae(self, verbose):

        es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        mcp = ModelCheckpoint(filepath=os.path.join(self.model_store, 'weights.h5'), monitor='val_loss', verbose=1,
                              save_best_only=True, save_weights_only=True)
        tb = TensorBoard(os.path.join(self.model_store, 'logs'))
        self.ae.fit(self.x_train, self.x_train, shuffle=True, epochs=200, callbacks=[es, rlr, mcp, tb],
                    validation_split=0.2, verbose=verbose, batch_size=64)

    def create_recon_images(self,num_recons):

        for l in range(0,num_recons):
            image_index = np.random.randint(0, len(self.x_test))
            xtm = (self.x_train_mean*self.subtract_pixel_mean).reshape(self.size_x,self.size_y,1)
            image_to_be_recon = np.uint8((self.x_test[image_index]+xtm)*255).reshape(self.size_y,self.size_x)
            image_to_be_recon_shaped = np.expand_dims(self.x_test[image_index], axis=0)
            class_of_image = self.y_test[image_index]

            reconstruction = np.uint8((self.ae.predict(image_to_be_recon_shaped)[0]+xtm)*255).reshape(self.size_y,self.size_x)

            encoding = self.encoder.predict(image_to_be_recon_shaped)[0]

            # print "########################"
            # print "ENCODING: ", encoding
            # print "########################"

            n = encoding.shape[0]
            samples = np.ones((7,n))*encoding + np.random.normal(0.0,0.1,size=(7,n))

            # print "########################"
            # print "SAMPLES: ", samples
            # print "########################"


            predictions_samples = np.uint8((self.decoder.predict(samples)+xtm)*255)

            f, axarr = plt.subplots(3, 3)
            plt.suptitle('Recon-class :' + str(class_of_image))
            axarr[0, 0].imshow(image_to_be_recon)
            axarr[0, 0].set_title('Original Image')

            axarr[0, 1].imshow(reconstruction)
            axarr[0, 1].set_title('R-Image')

            for i in range(0, 3):
                for j in range(0,3):

                    if((i==0 and j==0) or (i==0 and j==1)):
                        continue

                    axarr[i,j].imshow(predictions_samples[i*3+j-2].reshape(self.size_y,self.size_x))
                    axarr[i,j].set_title('R-Sampling :' + str(i*3+j-2))


            plt.savefig(os.path.join(self.model_store, 'recon-set' + str(l)))
            plt.close()

    def create_tsne_plot(self,graph_name):

        tsne_obj = TSNE(n_components=2,init='pca',random_state=0,verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        test_encodings = self.encoder.predict(self.x_test)
        full_array_feats = np.vstack((train_encodings,test_encodings))
        full_array_labels = np.vstack((self.y_train,self.y_test))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"


        plt.scatter(x=train_tsne_embedding[:,0],y=train_tsne_embedding[:,1],c=colors,cmap=plt.get_cmap('Set3'),
                    alpha=0.4)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

class MNIST_model_cluster:

    loss_list_fm=[]
    loss_list_dec = []
    means = None
    initial_means = None
    list_mean_disp = []

    def __init__(self,model_store,size_y=32,size_x=32,n_channels=1,h_units=64,subtract_pixel_mean=False,loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-2,lr_model=1e-4):

        self.clustering_lr = clustering_lr
        self.batch_size=batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model


        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        #Get MNIST 10 data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.y_train = self.y_train.reshape(self.y_train.shape[0],1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0],1)

        self.x_train = np.pad(self.x_train,pad_width=((0,0),(2,2),(2,2)),mode='constant',constant_values=0)
        self.x_test = np.pad(self.x_test, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)

        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        self.x_train_mean = np.mean(self.x_train, axis=0)

        if subtract_pixel_mean:
            self.subtract_pixel_mean = 1
            self.x_train -= self.x_train_mean
            self.x_test -= self.x_train_mean
        else:
            self.subtract_pixel_mean = 0

        self.x_train,self.y_train = shuffle(self.x_train,self.y_train)
        self.x_test,self.y_test = shuffle(self.x_test,self.y_test)

        #Change Data format to channels_last
        img_rows = self.x_train.shape[1]
        img_cols = self.x_train.shape[2]
        channels = 1

        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, channels)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, channels)

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels))

        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(8, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 14x14

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(16, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 7x7

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(32, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 64)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x/8 , size_y/8 , 64))

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(32, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(16, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(8, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(8, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        if(self.subtract_pixel_mean):
            recon = Conv2D(n_channels, (3, 3), activation='tanh', padding='same')
        else:
            recon = Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')

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


        self.ae = Model(inputs=[inp], outputs=[ae18])
        adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if(loss=='mse'):
            self.loss_compile = 'mse'

        elif(loss=='dssim'):
            self.loss_compile = dssim
        elif(loss=='bce'):
            self.loss_compile = 'binary_crossentropy'

        self.ae.compile(optimizer=adam_ae, loss=self.loss_compile)


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

        self.decoder = Model(inputs=[dinp], outputs=[de18])
        adam_dec = Adam(lr=self.lr_model)
        self.decoder.compile(optimizer=adam_dec,loss=self.loss_compile)

        if(os.path.isfile(os.path.join(model_store,'weights.h5'))):
            print "LOADING MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store,'weights.h5'))

    def make_trainable(self,net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def fit_model_ae(self,verbose=1,n_initial_epochs=10,n_train_iterations=1e5):

        #start_two_epoch_of_training
        self.ae.fit(self.x_train, self.x_train, shuffle=True, epochs=n_initial_epochs, batch_size=64, verbose=verbose)

        predicted_feats = self.encoder.predict(self.x_train)

        #Get means of predicted features
        km = KMeans(n_clusters=self.n_clusters, verbose=0, n_jobs=-1)
        km.fit(predicted_feats)

        self.means = np.copy(km.cluster_centers_)
        self.initial_means = np.copy(km.cluster_centers_)

        del km


        for i in tqdm(range(0, int(n_train_iterations))):

            #Get batch
            batch_idx=np.random.randint(0,len(self.x_train),size=self.batch_size)
            batch = self.x_train[batch_idx]

            #Get cluster-feats
            encoded_feats = self.encoder.predict(batch)
            cluster_assigns = self.get_assigns(self.means,encoded_feats)

            means_pre = np.copy(self.means)

            #Update means
            for j in range(0,self.n_clusters):
                feats_belonging_to_cluster = encoded_feats[cluster_assigns==j]
                if(feats_belonging_to_cluster.shape[0]>0):
                    mean_fbc = np.mean(feats_belonging_to_cluster,axis=0)
                    lr = self.clustering_lr/(i+1)
                    self.means[j]=(1-lr)*self.means[j]+lr*mean_fbc

            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means-means_pre),axis=1)))

            decoder_feats = self.means[cluster_assigns] + np.random.normal(loc=0.0,scale=np.var(np.var(encoded_feats,axis=0)),size=encoded_feats.shape)

            #Train decoder
            self.loss_list_dec.append(self.decoder.train_on_batch(decoder_feats, batch))
            #Fix_decoder
            self.make_trainable(self.decoder,False)


            #Train Full model
            self.loss_list_fm.append(self.ae.train_on_batch(batch,batch))
            if(np.random.rand()<0.3):
                n_batch_idx = np.random.randint(0, len(self.x_train), size=self.batch_size)
                n_batch = self.x_train[n_batch_idx]
                self.loss_list_fm.append(self.ae.train_on_batch(n_batch, n_batch))


            #Release decoder
            self.make_trainable(self.decoder,True)


        self.ae.save_weights(filepath=os.path.join(self.model_store,'weights.h5'))


    def get_assigns(self,means,feats):

        dist_matrix = cdist(feats,means)
        assigns = np.argmin(dist_matrix,axis=1)
        return assigns

    def create_recon_images(self,num_recons):

        for l in range(0,num_recons):
            image_index = np.random.randint(0, len(self.x_test))
            xtm = (self.x_train_mean*self.subtract_pixel_mean).reshape(self.size_x,self.size_y,1)
            image_to_be_recon = np.uint8((self.x_test[image_index]+xtm)*255).reshape(self.size_y,self.size_x)
            image_to_be_recon_shaped = np.expand_dims(self.x_test[image_index], axis=0)
            class_of_image = self.y_test[image_index]

            reconstruction = np.uint8((self.ae.predict(image_to_be_recon_shaped)[0]+xtm)*255).reshape(self.size_y,self.size_x)

            encoding = self.encoder.predict(image_to_be_recon_shaped)[0]

            # print "########################"
            # print "ENCODING: ", encoding
            # print "########################"

            n = encoding.shape[0]
            samples = np.ones((7,n))*encoding + np.random.normal(0.0,0.1,size=(7,n))

            # print "########################"
            # print "SAMPLES: ", samples
            # print "########################"


            predictions_samples = np.uint8((self.decoder.predict(samples)+xtm)*255)

            f, axarr = plt.subplots(3, 3)
            plt.suptitle('Recon-class :' + str(class_of_image))
            axarr[0, 0].imshow(image_to_be_recon)
            axarr[0, 0].set_title('Original Image')

            axarr[0, 1].imshow(reconstruction)
            axarr[0, 1].set_title('R-Image')

            for i in range(0, 3):
                for j in range(0,3):

                    if((i==0 and j==0) or (i==0 and j==1)):
                        continue

                    axarr[i,j].imshow(predictions_samples[i*3+j-2].reshape(self.size_y,self.size_x))
                    axarr[i,j].set_title('R-Sampling :' + str(i*3+j-2))


            plt.savefig(os.path.join(self.model_store, 'recon-set' + str(l)))
            plt.close()

    def generate_loss_graph(self,graph_name):

        hfm, = plt.plot(self.loss_list_fm,label='full_model')
        hdec, = plt.plot(self.loss_list_dec,label='decoder')
        plt.legend(handles=[hfm,hdec])
        plt.title('Losses per iteration')
        plt.xlabel('Iteration Index')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot(self,graph_name):

        tsne_obj = TSNE(n_components=2,init='pca',random_state=0,verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        test_encodings = self.encoder.predict(self.x_test)
        means = self.means
        full_array_feats = np.vstack((train_encodings,test_encodings,means))
        full_array_labels = np.vstack((self.y_train,self.y_test,np.ones((self.n_clusters,1))*10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"


        plt.scatter(x=train_tsne_embedding[:,0],y=train_tsne_embedding[:,1],c=colors,cmap=plt.get_cmap('Set3'),
                    alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def mean_displacement_distance(self):

        difference = self.means-self.initial_means

        print "####################################"
        print "Mean difference : "
        print  difference
        print "####################################"

        square = np.square(difference)
        sum_squares = np.sum(square,axis=1)

        print "####################################"
        print "Sum squares : "
        print  sum_squares
        print "####################################"


        displacement = np.sqrt(sum_squares)
        print "####################################"
        print "Mean displacement : ", displacement
        print "####################################"

    def generate_mean_displacement_graph(self,graph_name):

        mean_displacement = np.array(self.list_mean_disp)
        print mean_displacement.shape
        list_labels=[]

        for i in range(0,mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:,i],label='cluster_'+str(i+1))
            list_labels.append(x)

        plt.legend(handles=list_labels)
        plt.title('Mean displacements over iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Displacement Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

class MNIST_model_kmeans:

    means = None
    initial_means = None
    mean_assigns = None
    list_mean_disp = []
    cl_loss_weight = 0
    loss_list = []

    def __init__(self,model_store,size_y=32,size_x=32,n_channels=1,h_units=64,subtract_pixel_mean=False,loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-2,lr_model=1e-4,lamda=0.01,digits=True):

        self.clustering_lr = clustering_lr
        self.batch_size=batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model
        self.lamda = lamda
        self.means = np.zeros((n_clusters,h_units))

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        #Get MNIST 10 data
        if(digits):
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        y_train = y_train.reshape(y_train.shape[0],1)
        y_test = y_test.reshape(y_test.shape[0],1)

        x_train = np.pad(x_train,pad_width=((0,0),(2,2),(2,2)),mode='constant',constant_values=0)
        x_test = np.pad(x_test, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)

        self.x_train = np.vstack((x_train,x_test))
        self.y_train = np.vstack((y_train,y_test))

        self.x_train = self.x_train.astype('float32') / 255


        # If subtract pixel mean is enabled
        self.x_train_mean = np.mean(self.x_train, axis=0)

        if subtract_pixel_mean:
            self.subtract_pixel_mean = 1
            self.x_train -= self.x_train_mean
        else:
            self.subtract_pixel_mean = 0

        self.x_train,self.y_train = shuffle(self.x_train,self.y_train)

        #Change Data format to channels_last
        img_rows = self.x_train.shape[1]
        img_cols = self.x_train.shape[2]
        channels = 1

        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, channels)

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels))

        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(8, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 14x14

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(16, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 7x7

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(32, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 64)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x/8 , size_y/8 , 64))

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(32, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(16, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(8, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(8, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        if(self.subtract_pixel_mean):
            recon = Conv2D(n_channels, (3, 3), activation='tanh', padding='same')
        else:
            recon = Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')

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



        adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if(loss=='mse'):
            self.loss_fn = mean_squared_error
        elif(loss=='dssim'):
            self.loss_fn = dssim
        elif(loss=='bce'):
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

        self.y_obj = CustomClusterLayer(self.loss_fn,self.means,self.lamda,self.n_clusters,h_units)

        y = self.y_obj([inp, ae18, encoder])

        self.decoder = Model(inputs=[dinp], outputs=[de18])

        self.ae = Model(inputs=[inp], outputs=[y])

        self.ae.compile(optimizer=adam_ae, loss=None)

        if(os.path.isfile(os.path.join(model_store,'weights.h5'))):
            print "LOADING MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store,'weights.h5'))

    def fit_model_ae(self, verbose=1, n_initial_epochs=10, n_train_epochs=10, earlystopping=False, patience=10, least_loss=1e-5):

        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        # start_initial epoch_of_training
        self.ae.fit(self.x_train, shuffle=True, epochs=n_initial_epochs, batch_size=64, verbose=verbose)

        predicted_feats = self.encoder.predict(self.x_train)

        # Get means of predicted features
        km = KMeans(n_clusters=self.n_clusters, verbose=0, n_jobs=-1)
        km.fit(predicted_feats)

        self.means = np.copy(km.cluster_centers_)
        print "$$$$$$$$$$$$$$$$$$"
        print "MEANS_INITIAL"
        print "$$$$$$$$$$$$$$$$$$"
        print self.means

        self.initial_means = np.copy(km.cluster_centers_)

        loss_track = 0
        lowest_loss_ever = 1000.0

        del km

        # Make clustering loss weight to 1
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(1.0))
        K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

        for i in range(0, n_train_epochs):

            print (i + 1), "/", n_train_epochs, ":"
            history = self.ae.fit(self.x_train, batch_size=self.batch_size, epochs=1, verbose=verbose, shuffle=True)
            current_loss = history.history['loss'][0]
            self.loss_list.append(history.history['loss'][0])

            feats = self.encoder.predict(self.x_train)
            cluster_assigns = self.get_assigns(self.means, feats)

            means_pre = np.copy(self.means)
            self.update_means(feats, cluster_assigns)
            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

            K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

            if (lowest_loss_ever - current_loss > least_loss):
                loss_track = 0
                print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                lowest_loss_ever = current_loss
            else:
                print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                loss_track += 1
                print "Loss track is :", loss_track, "/", patience

            if (earlystopping and loss_track > patience):
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "EARLY STOPPING AT EPOCH :", i
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                break

        self.ae.save_weights(filepath=os.path.join(self.model_store, 'weights.h5'))

    def get_assigns(self,means,feats):

        dist_matrix = cdist(feats,means)
        assigns = np.argmin(dist_matrix,axis=1)
        return assigns

    def update_means(self,feats,cluster_assigns):

        ck = np.zeros((self.means.shape[0]))
        for idx,i in enumerate(feats):
            ck[cluster_assigns[idx]]+=1
            cki = ck[cluster_assigns[idx]]
            self.means[cluster_assigns[idx]] -= (1/cki)*(self.means[cluster_assigns[idx]]-feats[idx])*self.clustering_lr

    def create_recon_images(self,num_recons):

        for l in range(0,num_recons):
            image_index = np.random.randint(0, len(self.x_train))
            xtm = (self.x_train_mean*self.subtract_pixel_mean).reshape(self.size_x,self.size_y,1)
            image_to_be_recon = np.uint8((self.x_train[image_index]+xtm)*255).reshape(self.size_y,self.size_x)
            image_to_be_recon_shaped = np.expand_dims(self.x_train[image_index], axis=0)
            class_of_image = self.y_train[image_index]

            reconstruction = np.uint8((self.ae.predict(image_to_be_recon_shaped)[0]+xtm)*255).reshape(self.size_y,self.size_x)

            encoding = self.encoder.predict(image_to_be_recon_shaped)[0]


            n = encoding.shape[0]
            samples = np.ones((7,n))*encoding + np.random.normal(0.0,0.1,size=(7,n))


            predictions_samples = np.uint8((self.decoder.predict(samples)+xtm)*255)

            f, axarr = plt.subplots(3, 3)
            plt.suptitle('Recon-class :' + str(class_of_image))
            axarr[0, 0].imshow(image_to_be_recon)
            axarr[0, 0].set_title('Original Image')

            axarr[0, 1].imshow(reconstruction)
            axarr[0, 1].set_title('R-Image')

            for i in range(0, 3):
                for j in range(0,3):

                    if((i==0 and j==0) or (i==0 and j==1)):
                        continue

                    axarr[i,j].imshow(predictions_samples[i*3+j-2].reshape(self.size_y,self.size_x))
                    axarr[i,j].set_title('R-Sample :' + str(i*3+j-2))


            plt.savefig(os.path.join(self.model_store, 'recon-set' + str(l)))
            plt.close()

    def generate_loss_graph(self,graph_name):

        hfm, = plt.plot(self.loss_list,label='loss')
        plt.legend(handles=[hfm])
        plt.title('Losses per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot(self,graph_name):

        tsne_obj = TSNE(n_components=2,init='pca',random_state=0,verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        means = self.means
        full_array_feats = np.vstack((train_encodings,means))
        full_array_labels = np.vstack((self.y_train,np.ones((self.n_clusters,1))*10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"


        plt.scatter(x=train_tsne_embedding[:,0],y=train_tsne_embedding[:,1],c=colors,cmap=plt.get_cmap('Set3'),
                    alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot3d(self, graph_name):

        tsne_obj = TSNE(n_components=3, init='pca', random_state=0, verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        means = self.means
        full_array_feats = np.vstack((train_encodings, means))
        full_array_labels = np.vstack((self.y_train, np.ones((self.n_clusters, 1)) * 10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING 3D"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_tsne_embedding[:, 0], train_tsne_embedding[:, 1], train_tsne_embedding[:, 2], c=colors,
                   cmap=plt.get_cmap('Set3'),alpha=0.6)
        ax.set_title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        ax.legend()

        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def mean_displacement_distance(self):

        difference = self.means-self.initial_means

        print "####################################"
        print "Mean difference : "
        print  difference
        print "####################################"

        square = np.square(difference)
        sum_squares = np.sum(square,axis=1)

        print "####################################"
        print "Sum squares : "
        print  sum_squares
        print "####################################"


        displacement = np.sqrt(sum_squares)
        print "####################################"
        print "Mean displacement : ", displacement
        print "####################################"

    def generate_mean_displacement_graph(self,graph_name):

        mean_displacement = np.array(self.list_mean_disp)
        print mean_displacement.shape
        list_labels=[]

        for i in range(0,mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:,i],label='cluster_'+str(i+1))
            list_labels.append(x)

        plt.legend(handles=list_labels)
        plt.title('Mean displacements over iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Displacement Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def generate_assignment_graph(self,graph_name):

        feats = self.encoder.predict(self.x_train)
        assigns = self.get_assigns(self.means,feats)
        list_assigns=[]

        for i in range(0,self.n_clusters):
            list_assigns.append(np.sum(assigns==i))

        plt.bar(x=range(0,self.n_clusters),height=list_assigns,width=0.25)
        plt.title('Cluster Assignments')
        plt.xlabel('Cluster ids')
        plt.ylabel('Number of assignments')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def decode_means(self,graph_name):

        xtm = (self.x_train_mean * self.subtract_pixel_mean).reshape(self.size_x, self.size_y, 1)

        means_decoded = np.uint8((self.decoder.predict(self.means) + xtm) * 255)

        f, axarr = plt.subplots(2, 5)

        plt.suptitle('Means decoded')

        for i in range(0, 2):
            for j in range(0, 5):
                axarr[i, j].imshow(means_decoded[i*5+j].reshape(self.size_y,self.size_x))
                axarr[i, j].set_title('Mean :' + str(i*5+j))

        plt.savefig(os.path.join(self.model_store,graph_name))
        plt.close()

class CIFAR_model_kmeans:


    means = None
    initial_means = None
    mean_assigns = None
    list_mean_disp = []
    cl_loss_weight = 0
    loss_list = []

    def __init__(self,model_store,size_y=32,size_x=32,n_channels=3,h_units=256,subtract_pixel_mean=False,loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-2,lr_model=1e-4,lamda=0.01):

        self.clustering_lr = clustering_lr
        self.batch_size=batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model
        self.lamda = lamda
        self.means = np.zeros((n_clusters,h_units))

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        #Get MNIST 10 data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        y_train = y_train.reshape(y_train.shape[0],1)
        y_test = y_test.reshape(y_test.shape[0],1)


        self.x_train = np.vstack((x_train,x_test))
        self.y_train = np.vstack((y_train,y_test))

        self.x_train = self.x_train.astype('float32') / 255


        # If subtract pixel mean is enabled
        self.x_train_mean = np.mean(self.x_train, axis=0)

        if subtract_pixel_mean:
            self.subtract_pixel_mean = 1
            self.x_train -= self.x_train_mean
        else:
            self.subtract_pixel_mean = 0

        self.x_train,self.y_train = shuffle(self.x_train,self.y_train)

        #Change Data format to channels_last
        img_rows = self.x_train.shape[1]
        img_cols = self.x_train.shape[2]
        channels = n_channels

        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, channels)

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels))

        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(64, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 14x14

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(128, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 7x7

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(256, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 512)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x/8 , size_y/8 , 512))

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(256, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(128, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(64, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(32, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        if(self.subtract_pixel_mean):
            recon = Conv2D(n_channels, (3, 3), activation='tanh', padding='same')
        else:
            recon = Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')

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



        adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if(loss=='mse'):
            self.loss_fn = mean_squared_error
        elif(loss=='dssim'):
            self.loss_fn = dssim
        elif(loss=='bce'):
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

        self.y_obj = CustomClusterLayer(self.loss_fn,self.means,self.lamda,self.n_clusters,h_units)

        y = self.y_obj([inp, ae18, encoder])

        self.decoder = Model(inputs=[dinp], outputs=[de18])

        self.ae = Model(inputs=[inp], outputs=[y])

        self.ae.compile(optimizer=adam_ae, loss=None)

        if(os.path.isfile(os.path.join(model_store,'weights.h5'))):
            print "LOADING MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store,'weights.h5'))

    def fit_model_ae(self, verbose=1, n_initial_epochs=10, n_train_epochs=10, earlystopping=False, patience=10, least_loss=1e-5):

        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        # start_initial epoch_of_training
        self.ae.fit(self.x_train, shuffle=True, epochs=n_initial_epochs, batch_size=64, verbose=verbose)

        predicted_feats = self.encoder.predict(self.x_train)

        # Get means of predicted features
        km = KMeans(n_clusters=self.n_clusters, verbose=0, n_jobs=-1)
        km.fit(predicted_feats)

        self.means = np.copy(km.cluster_centers_)
        print "$$$$$$$$$$$$$$$$$$"
        print "MEANS_INITIAL"
        print "$$$$$$$$$$$$$$$$$$"
        print self.means

        self.initial_means = np.copy(km.cluster_centers_)

        loss_track = 0
        lowest_loss_ever = 1000.0

        del km

        # Make clustering loss weight to 1
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(1.0))
        K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

        for i in range(0, n_train_epochs):

            print (i + 1), "/", n_train_epochs, ":"
            history = self.ae.fit(self.x_train, batch_size=self.batch_size, epochs=1, verbose=verbose, shuffle=True)
            current_loss = history.history['loss'][0]
            self.loss_list.append(history.history['loss'][0])

            feats = self.encoder.predict(self.x_train)
            cluster_assigns = self.get_assigns(self.means, feats)

            means_pre = np.copy(self.means)
            self.update_means(feats, cluster_assigns)
            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

            K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

            if (lowest_loss_ever - current_loss > least_loss):
                loss_track = 0
                print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                lowest_loss_ever = current_loss
            else:
                print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                loss_track += 1
                print "Loss track is :", loss_track, "/", patience

            if (earlystopping and loss_track > patience):
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "EARLY STOPPING AT EPOCH :", i
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                break

        self.ae.save_weights(filepath=os.path.join(self.model_store, 'weights.h5'))


    def get_assigns(self,means,feats):

        dist_matrix = cdist(feats,means)
        assigns = np.argmin(dist_matrix,axis=1)
        return assigns

    def update_means(self,feats,cluster_assigns):

        ck = np.zeros((self.means.shape[0]))
        for idx,i in enumerate(feats):
            ck[cluster_assigns[idx]]+=1
            cki = ck[cluster_assigns[idx]]
            self.means[cluster_assigns[idx]] -= (1/cki)*(self.means[cluster_assigns[idx]]-feats[idx])*self.clustering_lr

    def create_recon_images(self, num_recons):

        for l in range(0, num_recons):
            image_index = np.random.randint(0, len(self.x_train))
            xtm = self.x_train_mean * self.subtract_pixel_mean
            image_to_be_recon = np.uint8((self.x_train[image_index] + xtm) * 255)
            image_to_be_recon_shaped = np.expand_dims(self.x_train[image_index], axis=0)
            class_of_image = self.y_train[image_index]

            reconstruction = np.uint8((self.ae.predict(image_to_be_recon_shaped)[0] + xtm) * 255)
            encoding = self.encoder.predict(image_to_be_recon_shaped)[0]

            # print "########################"
            # print "ENCODING: ", encoding
            # print "########################"

            n = encoding.shape[0]
            samples = np.ones((7, n)) * encoding + np.random.normal(0.0, 0.1, size=(7, n))

            # print "########################"
            # print "SAMPLES: ", samples
            # print "########################"

            predictions_samples = np.uint8((self.decoder.predict(samples) + xtm) * 255)

            f, axarr = plt.subplots(3, 3)
            plt.suptitle('Recon-class :' + str(class_of_image))
            axarr[0, 0].imshow(image_to_be_recon)
            axarr[0, 0].set_title('Original Image')

            axarr[0, 1].imshow(reconstruction)
            axarr[0, 1].set_title('R-Image')

            for i in range(0, 3):
                for j in range(0, 3):

                    if ((i == 0 and j == 0) or (i == 0 and j == 1)):
                        continue

                    axarr[i, j].imshow(predictions_samples[i * 3 + j - 2])
                    axarr[i, j].set_title('R-Sampling :' + str(i * 3 + j - 2))

            plt.savefig(os.path.join(self.model_store, 'recon-set' + str(l)))
            plt.close()

    def generate_loss_graph(self,graph_name):

        hfm, = plt.plot(self.loss_list,label='loss')
        plt.legend(handles=[hfm])
        plt.title('Losses per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot(self,graph_name):

        tsne_obj = TSNE(n_components=2,init='pca',random_state=0,verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        means = self.means
        full_array_feats = np.vstack((train_encodings,means))
        full_array_labels = np.vstack((self.y_train,np.ones((self.n_clusters,1))*10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"


        plt.scatter(x=train_tsne_embedding[:,0],y=train_tsne_embedding[:,1],c=colors,cmap=plt.get_cmap('Set3'),
                    alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot3d(self, graph_name):

        tsne_obj = TSNE(n_components=3, init='pca', random_state=0, verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        means = self.means
        full_array_feats = np.vstack((train_encodings, means))
        full_array_labels = np.vstack((self.y_train, np.ones((self.n_clusters, 1)) * 10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING 3D"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_tsne_embedding[:, 0], train_tsne_embedding[:, 1], train_tsne_embedding[:, 2], c=colors,
                   cmap=plt.get_cmap('Set3'),alpha=0.6)
        ax.set_title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        ax.legend()

        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def mean_displacement_distance(self):

        difference = self.means-self.initial_means

        print "####################################"
        print "Mean difference : "
        print  difference
        print "####################################"

        square = np.square(difference)
        sum_squares = np.sum(square,axis=1)

        print "####################################"
        print "Sum squares : "
        print  sum_squares
        print "####################################"


        displacement = np.sqrt(sum_squares)
        print "####################################"
        print "Mean displacement : ", displacement
        print "####################################"

    def generate_mean_displacement_graph(self,graph_name):

        mean_displacement = np.array(self.list_mean_disp)
        print mean_displacement.shape
        list_labels=[]

        for i in range(0,mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:,i],label='cluster_'+str(i+1))
            list_labels.append(x)

        plt.legend(handles=list_labels)
        plt.title('Mean displacements over iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Displacement Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def generate_assignment_graph(self,graph_name):

        feats = self.encoder.predict(self.x_train)
        assigns = self.get_assigns(self.means,feats)
        list_assigns = []

        for i in range(0,self.n_clusters):
            list_assigns.append(np.sum(assigns == i))

        plt.bar(x=range(0,self.n_clusters),height=list_assigns,width=0.25)
        plt.title('Cluster Assignments')
        plt.xlabel('Cluster ids')
        plt.ylabel('Number of assignments')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def decode_means(self,graph_name):

        xtm = self.x_train_mean * self.subtract_pixel_mean
        means_decoded = np.uint8((self.decoder.predict(self.means) + xtm) * 255)
        f, axarr = plt.subplots(2, 5)
        plt.suptitle('Means decoded')

        for i in range(0, 2):
            for j in range(0, 5):
                axarr[i, j].imshow(means_decoded[i*5+j])
                axarr[i, j].set_title('Mean :' + str(i*5+j))

        plt.savefig(os.path.join(self.model_store,graph_name))
        plt.close()

class Artificial_Data_Test:
    means = None
    initial_means = None
    mean_assigns = None
    list_mean_disp = []
    cl_loss_weight = 0
    loss_list = []

    def __init__(self, model_store, n_samples_each = 10000, proj_dim = 32, loss='mse', batch_size=64, clustering_lr=1e-2, lr_model=1e-4, lamda=0.01,assignment_lamda=0.01):

        self.clustering_lr = clustering_lr
        self.batch_size = batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = 4
        self.lr_model = lr_model
        self.lamda = lamda
        self.assignment_lamda = assignment_lamda
        self.h_units = 2
        self.proj_dim = proj_dim
        self.n_samples_each = n_samples_each
        self.means = np.zeros((self.n_clusters, self.h_units))
        self.cluster_assigns = None


        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        points_list = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
        self.points = np.array(points_list)
        cov_matrix = np.array([[0.001, 0], [0, 0.001]])

        # Get 2D data
        if (not os.path.isfile(os.path.join(model_store, 'samples.npy'))):

            samples_list =[]
            samples_assignments_list = []

            for idx,i in enumerate(self.points):
                for j in range(0,n_samples_each):
                    samples_list.append(np.random.multivariate_normal(i,cov=cov_matrix))
                    samples_assignments_list.append(idx)

            self.samples = np.array(samples_list)
            np.save(os.path.join(model_store,'samples.npy'),self.samples)
            self.assignments = np.array(samples_assignments_list)
            np.save(os.path.join(model_store,'assignments.npy'),self.assignments)

        else:
            self.samples = np.load(os.path.join(model_store,'samples.npy'))
            self.assignments = np.load(os.path.join(model_store,'assignments.npy'))


        self.samples = (self.samples).astype('float32')

        inp_p_model = Input(shape=(2,))
        proj_model = Dense(units=(proj_dim/2),activation='tanh')(inp_p_model)
        proj_model = Dense(units=proj_dim,activation='sigmoid')(proj_model)

        self.projector = Model(inputs=[inp_p_model],outputs=[proj_model])

        if (os.path.isfile(os.path.join(model_store, 'projector_weights.h5'))):
            print "LOADING PROJECTOR MODEL WEIGHTS FROM WEIGHTS FILE"
            self.projector.load_weights(os.path.join(model_store, 'projector_weights.h5'))
        else:
            self.projector.save_weights(filepath=os.path.join(self.model_store, 'projector_weights.h5'))

        self.x_train = self.projector.predict(self.samples)


        # MODEL CREATION
        inp = Input(shape=(proj_dim,))
        x1 = Dense(units=proj_dim/2,activation='tanh')(inp)

        encoder = Dense(units=self.h_units, activation='sigmoid')(x1)

        dec1 = Dense(units=proj_dim/2,activation='tanh')
        recon = Dense(units=proj_dim,activation='sigmoid')


        ae1 = dec1(encoder)
        ae2 = recon(ae1)

        adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if (loss == 'mse'):
            self.loss_fn = mean_squared_error
        elif (loss == 'dssim'):
            self.loss_fn = dssim
        elif (loss == 'bce'):
            self.loss_fn = binary_crossentropy

        self.encoder = Model(inputs=[inp], outputs=[encoder])

        dinp = Input(shape=(self.h_units,))
        de1 = dec1(dinp)
        de2 = recon(de1)
        inp_assignments = Input(shape=(1,))

        self.y_obj = CustomClusterLayer_Test(self.loss_fn, self.means, self.lamda, self.n_clusters, self.h_units,self.assignment_lamda)

        y = self.y_obj([inp, ae2, encoder,inp_assignments])

        self.decoder = Model(inputs=[dinp], outputs=[de2])

        self.ae = Model(inputs=[inp,inp_assignments], outputs=[y])

        self.ae.compile(optimizer=adam_ae, loss=None)

        if (os.path.isfile(os.path.join(model_store, 'weights.h5'))):
            print "LOADING MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store, 'weights.h5'))

    def fit_model_ae(self, verbose=1, n_initial_epochs=10, n_train_epochs=10, earlystopping=False, patience=10,
                     least_loss=1e-5):

        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        # start_initial epoch_of_training
        feats = self.encoder.predict(self.x_train)
        self.cluster_assigns = self.get_assigns(self.means,feats)

        self.ae.fit([self.x_train,self.cluster_assigns], shuffle=True, epochs=n_initial_epochs, batch_size=64, verbose=verbose)

        feats = self.encoder.predict(self.x_train)

        # Get means of predicted features
        km = KMeans(n_clusters=self.n_clusters, verbose=0, n_jobs=-1)
        km.fit(feats)

        self.means = np.copy(km.cluster_centers_).astype('float64')
        self.cluster_assigns = self.get_assigns(self.means, feats)

        self.create_2d_encoding_plot('after_pre_train.png')
        print "$$$$$$$$$$$$$$$$$$"
        print "MEANS_INITIAL"
        print "$$$$$$$$$$$$$$$$$$"
        print self.means

        self.initial_means = np.copy(km.cluster_centers_)

        loss_track = 0
        lowest_loss_ever = 1000.0

        del km

        feats = self.encoder.predict(self.x_train)
        self.cluster_assigns = self.get_assigns(self.means,feats)

        # Make clustering loss weight to 1
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(1.0))
        K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))


        for i in range(0, n_train_epochs):

            print (i + 1), "/", n_train_epochs, ":"
            history = self.ae.fit([self.x_train,self.cluster_assigns], batch_size=self.batch_size, epochs=1, verbose=verbose, shuffle=True)
            current_loss = history.history['loss'][0]
            self.loss_list.append(history.history['loss'][0])

            feats = self.encoder.predict(self.x_train)
            self.cluster_assigns = self.get_assigns(self.means, feats)

            means_pre = np.copy(self.means)
            self.update_means(feats, self.cluster_assigns)

            #update cluster assigns for the next loop
            self.cluster_assigns = self.get_assigns(self.means, feats)

            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

            K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

            self.create_2d_encoding_plot('after_train_' + str(i).zfill(3) + '.png')


            if (lowest_loss_ever - current_loss > least_loss):
                loss_track = 0
                print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                lowest_loss_ever = current_loss
            else:
                print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                loss_track += 1
                print "Loss track is :", loss_track, "/", patience

            if (earlystopping and loss_track > patience):
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "EARLY STOPPING AT EPOCH :", i
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                break

        self.ae.save_weights(filepath=os.path.join(self.model_store, 'weights.h5'))

    def get_assigns(self, means, feats):

        dist_matrix = cdist(feats, means)
        assigns = np.argmin(dist_matrix, axis=1)
        return assigns

    def update_means(self, feats, cluster_assigns):

        # for i in range(0,self.n_clusters):
        #     feats_i = feats[np.where(cluster_assigns==i)]
        #     if(feats_i.shape[0]>0):
        #         new_mean = np.mean(feats_i,axis=0)
        #         self.means[i]=new_mean
        #         # self.means[i] = self.means[i]*(1-self.clustering_lr) + (new_mean)*self.clustering_lr

        ck = np.zeros((self.means.shape[0]))
        for idx, i in enumerate(feats):
            ck[cluster_assigns[idx]] += 1
            cki = ck[cluster_assigns[idx]]
            self.means[cluster_assigns[idx]] -= (1 / cki) * (self.means[cluster_assigns[idx]] - feats[idx]) * self.clustering_lr

    def generate_loss_graph(self, graph_name):

        hfm, = plt.plot(self.loss_list, label='loss')
        plt.legend(handles=[hfm])
        plt.title('Losses per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def create_2d_encoding_plot(self, graph_name):

        train_encodings = self.encoder.predict(self.x_train)
        means = self.means
        full_array_feats = np.vstack((train_encodings, means))

        full_array_labels0 = np.vstack((self.cluster_assigns.reshape(self.cluster_assigns.shape[0],1), np.ones((self.n_clusters,1)) * 5))
        colors0 = full_array_labels0.reshape((full_array_labels0.shape[0],))

        full_array_labels1 = np.vstack((self.assignments.reshape(self.assignments.shape[0],1), np.ones((self.n_clusters,1)) * 5))
        colors1 = full_array_labels1.reshape((full_array_labels1.shape[0],))

        train_tsne_embedding = full_array_feats

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        ax1.scatter(x=train_tsne_embedding[:, 0], y=train_tsne_embedding[:, 1], c=colors0,alpha=0.6)
        ax1.set_title('Cluster assigns-model')

        ax2.scatter(x=train_tsne_embedding[:, 0], y=train_tsne_embedding[:, 1], c=colors1,alpha=0.6)
        ax2.set_title('Cluster assigns-data')

        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
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

        for i in range(0, mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:, i], label='cluster_' + str(i + 1))
            list_labels.append(x)

        plt.legend(handles=list_labels)
        plt.title('Mean displacements over iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Displacement Value')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def generate_assignment_graph(self, graph_name):

        feats = self.encoder.predict(self.x_train)
        assigns = self.get_assigns(self.means, feats)
        list_assigns = []

        for i in range(0, self.n_clusters):
            list_assigns.append(np.sum(assigns == i))

        plt.bar(x=range(0, self.n_clusters), height=list_assigns, width=0.25)
        plt.title('Cluster Assignments')
        plt.xlabel('Cluster ids')
        plt.ylabel('Number of assignments')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def plot_data_2d(self, graph_name):

        train_encodings = self.samples
        means = self.points
        full_array_feats = np.vstack((train_encodings, means))
        full_array_labels = np.vstack((self.assignments.reshape(self.assignments.shape[0],1), np.ones((self.n_clusters,1)) * 5))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        train_tsne_embedding = full_array_feats

        plt.scatter(x=train_tsne_embedding[:, 0], y=train_tsne_embedding[:, 1], c=colors, alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

class MNIST_model_kmeans_assignments:

    means = None
    initial_means = None
    mean_assigns = None
    list_mean_disp = []
    cl_loss_weight = 0
    loss_list = []

    def __init__(self,model_store,size_y=32,size_x=32,n_channels=1,h_units=64,subtract_pixel_mean=False,loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-2,lr_model=1e-4,lamda=0.01,digits=True,lamda_assign=0.01):

        self.clustering_lr = clustering_lr
        self.batch_size=batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model
        self.lamda = lamda
        self.means = np.zeros((n_clusters,h_units))
        self.cluster_assigns = None
        self.assignment_lamda = lamda_assign

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        #Get MNIST 10 data
        if(digits):
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        y_train = y_train.reshape(y_train.shape[0],1)
        y_test = y_test.reshape(y_test.shape[0],1)

        x_train = np.pad(x_train,pad_width=((0,0),(2,2),(2,2)),mode='constant',constant_values=0)
        x_test = np.pad(x_test, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)

        self.x_train = np.vstack((x_train,x_test))
        self.y_train = np.vstack((y_train,y_test))

        self.x_train = self.x_train.astype('float32') / 255


        # If subtract pixel mean is enabled
        self.x_train_mean = np.mean(self.x_train, axis=0)

        if subtract_pixel_mean:
            self.subtract_pixel_mean = 1
            self.x_train -= self.x_train_mean
        else:
            self.subtract_pixel_mean = 0

        self.x_train,self.y_train = shuffle(self.x_train,self.y_train)

        #Change Data format to channels_last
        img_rows = self.x_train.shape[1]
        img_cols = self.x_train.shape[2]
        channels = 1

        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, channels)

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels))

        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(8, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 14x14

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(16, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 7x7

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(32, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 64)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x/8 , size_y/8 , 64))

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(32, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(16, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(8, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(8, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        if(self.subtract_pixel_mean):
            recon = Conv2D(n_channels, (3, 3), activation='tanh', padding='same')
        else:
            recon = Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')

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



        adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if(loss=='mse'):
            self.loss_fn = mean_squared_error
        elif(loss=='dssim'):
            self.loss_fn = dssim
        elif(loss=='bce'):
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

        inp_assignments = Input(shape=(1,))

        self.y_obj = CustomClusterLayer_Test(self.loss_fn,self.means,self.lamda,self.n_clusters,h_units,self.assignment_lamda)

        y = self.y_obj([inp, ae18, encoder,inp_assignments])

        self.decoder = Model(inputs=[dinp], outputs=[de18])

        self.ae = Model(inputs=[inp,inp_assignments], outputs=[y])

        self.ae.compile(optimizer=adam_ae, loss=None)

        if(os.path.isfile(os.path.join(model_store,'weights.h5'))):
            print "LOADING MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store,'weights.h5'))

    def fit_model_ae(self, verbose=1, n_initial_epochs=10, n_train_epochs=10, earlystopping=False, patience=10, least_loss=1e-5):

        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        # start_initial epoch_of_training
        feats = self.encoder.predict(self.x_train)
        self.cluster_assigns = self.get_assigns(self.means,feats)

        self.ae.fit([self.x_train,self.cluster_assigns], shuffle=True, epochs=n_initial_epochs, batch_size=64, verbose=verbose)

        feats = self.encoder.predict(self.x_train)

        # Get means of predicted features
        km = KMeans(n_clusters=self.n_clusters, verbose=0, n_jobs=-1)
        km.fit(feats)

        self.means = np.copy(km.cluster_centers_).astype('float64')
        self.cluster_assigns = self.get_assigns(self.means, feats)


        print "$$$$$$$$$$$$$$$$$$"
        print "MEANS_INITIAL"
        print "$$$$$$$$$$$$$$$$$$"
        print self.means

        self.initial_means = np.copy(km.cluster_centers_)

        loss_track = 0
        lowest_loss_ever = 1000.0

        del km

        feats = self.encoder.predict(self.x_train)
        self.cluster_assigns = self.get_assigns(self.means,feats)

        # Make clustering loss weight to 1
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(1.0))
        K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))


        for i in range(0, n_train_epochs):

            print (i + 1), "/", n_train_epochs, ":"
            history = self.ae.fit([self.x_train,self.cluster_assigns], batch_size=self.batch_size, epochs=1, verbose=verbose, shuffle=True)
            current_loss = history.history['loss'][0]
            self.loss_list.append(history.history['loss'][0])

            feats = self.encoder.predict(self.x_train)
            self.cluster_assigns = self.get_assigns(self.means, feats)

            means_pre = np.copy(self.means)
            self.update_means(feats, self.cluster_assigns)

            #update cluster assigns for the next loop
            self.cluster_assigns = self.get_assigns(self.means, feats)

            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

            K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

            if (lowest_loss_ever - current_loss > least_loss):
                loss_track = 0
                print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                lowest_loss_ever = current_loss
            else:
                print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                loss_track += 1
                print "Loss track is :", loss_track, "/", patience

            if (earlystopping and loss_track > patience):
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "EARLY STOPPING AT EPOCH :", i
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                break

        self.ae.save_weights(filepath=os.path.join(self.model_store, 'weights.h5'))

    def get_assigns(self,means,feats):

        dist_matrix = cdist(feats,means)
        assigns = np.argmin(dist_matrix,axis=1)
        return assigns

    def update_means(self,feats,cluster_assigns):

        ck = np.zeros((self.means.shape[0]))
        for idx,i in enumerate(feats):
            ck[cluster_assigns[idx]]+=1
            cki = ck[cluster_assigns[idx]]
            self.means[cluster_assigns[idx]] -= (1/cki)*(self.means[cluster_assigns[idx]]-feats[idx])*self.clustering_lr

    def create_recon_images(self,num_recons):

        for l in range(0,num_recons):
            image_index = np.random.randint(0, len(self.x_train))
            xtm = (self.x_train_mean*self.subtract_pixel_mean).reshape(self.size_x,self.size_y,1)
            image_to_be_recon = np.uint8((self.x_train[image_index]+xtm)*255).reshape(self.size_y,self.size_x)
            image_to_be_recon_shaped = np.expand_dims(self.x_train[image_index], axis=0)
            cl_assign = np.expand_dims(self.cluster_assigns[image_index], axis=0)
            class_of_image = self.y_train[image_index]

            reconstruction = np.uint8((self.ae.predict([image_to_be_recon_shaped,cl_assign])[0]+xtm)*255).reshape(self.size_y,self.size_x)

            encoding = self.encoder.predict(image_to_be_recon_shaped)[0]


            n = encoding.shape[0]
            samples = np.ones((7,n))*encoding + np.random.normal(0.0,0.1,size=(7,n))


            predictions_samples = np.uint8((self.decoder.predict(samples)+xtm)*255)

            f, axarr = plt.subplots(3, 3)
            plt.suptitle('Recon-class :' + str(class_of_image))
            axarr[0, 0].imshow(image_to_be_recon)
            axarr[0, 0].set_title('Original Image')

            axarr[0, 1].imshow(reconstruction)
            axarr[0, 1].set_title('R-Image')

            for i in range(0, 3):
                for j in range(0,3):

                    if((i==0 and j==0) or (i==0 and j==1)):
                        continue

                    axarr[i,j].imshow(predictions_samples[i*3+j-2].reshape(self.size_y,self.size_x))
                    axarr[i,j].set_title('R-Sample :' + str(i*3+j-2))


            plt.savefig(os.path.join(self.model_store, 'recon-set' + str(l)))
            plt.close()

    def generate_loss_graph(self,graph_name):

        hfm, = plt.plot(self.loss_list,label='loss')
        plt.legend(handles=[hfm])
        plt.title('Losses per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot(self,graph_name):

        tsne_obj = TSNE(n_components=2,init='pca',random_state=0,verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        means = self.means
        full_array_feats = np.vstack((train_encodings,means))
        full_array_labels = np.vstack((self.y_train,np.ones((self.n_clusters,1))*10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"


        plt.scatter(x=train_tsne_embedding[:,0],y=train_tsne_embedding[:,1],c=colors,cmap=plt.get_cmap('Set3'),
                    alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot3d(self, graph_name):

        tsne_obj = TSNE(n_components=3, init='pca', random_state=0, verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        means = self.means
        full_array_feats = np.vstack((train_encodings, means))
        full_array_labels = np.vstack((self.y_train, np.ones((self.n_clusters, 1)) * 10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING 3D"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_tsne_embedding[:, 0], train_tsne_embedding[:, 1], train_tsne_embedding[:, 2], c=colors,
                   cmap=plt.get_cmap('Set3'),alpha=0.6)
        ax.set_title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        ax.legend()

        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def mean_displacement_distance(self):

        difference = self.means-self.initial_means

        print "####################################"
        print "Mean difference : "
        print  difference
        print "####################################"

        square = np.square(difference)
        sum_squares = np.sum(square,axis=1)

        print "####################################"
        print "Sum squares : "
        print  sum_squares
        print "####################################"


        displacement = np.sqrt(sum_squares)
        print "####################################"
        print "Mean displacement : ", displacement
        print "####################################"

    def generate_mean_displacement_graph(self,graph_name):

        mean_displacement = np.array(self.list_mean_disp)
        print mean_displacement.shape
        list_labels=[]

        for i in range(0,mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:,i],label='cluster_'+str(i+1))
            list_labels.append(x)

        plt.legend(handles=list_labels)
        plt.title('Mean displacements over iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Displacement Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def generate_assignment_graph(self,graph_name):

        feats = self.encoder.predict(self.x_train)
        assigns = self.get_assigns(self.means,feats)
        list_assigns=[]

        for i in range(0,self.n_clusters):
            list_assigns.append(np.sum(assigns==i))

        plt.bar(x=range(0,self.n_clusters),height=list_assigns,width=0.25)
        plt.title('Cluster Assignments')
        plt.xlabel('Cluster ids')
        plt.ylabel('Number of assignments')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def decode_means(self,graph_name):

        xtm = (self.x_train_mean * self.subtract_pixel_mean).reshape(self.size_x, self.size_y, 1)

        means_decoded = np.uint8((self.decoder.predict(self.means) + xtm) * 255)

        f, axarr = plt.subplots(2, 5)

        plt.suptitle('Means decoded')

        for i in range(0, 2):
            for j in range(0, 5):
                axarr[i, j].imshow(means_decoded[i*5+j].reshape(self.size_y,self.size_x))
                axarr[i, j].set_title('Mean :' + str(i*5+j))

        plt.savefig(os.path.join(self.model_store,graph_name))
        plt.close()

class CIFAR_model_kmeans_assignments:


    means = None
    initial_means = None
    mean_assigns = None
    list_mean_disp = []
    cl_loss_weight = 0
    loss_list = []

    def __init__(self,model_store,size_y=32,size_x=32,n_channels=3,h_units=256,subtract_pixel_mean=False,loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-2,lr_model=1e-4,lamda=0.01,lamda_assign=0.01):

        self.clustering_lr = clustering_lr
        self.batch_size=batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model
        self.lamda = lamda
        self.means = np.zeros((n_clusters,h_units))
        self.cluster_assigns = None
        self.assignment_lamda = lamda_assign

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        #Get MNIST 10 data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        y_train = y_train.reshape(y_train.shape[0],1)
        y_test = y_test.reshape(y_test.shape[0],1)


        self.x_train = np.vstack((x_train,x_test))
        self.y_train = np.vstack((y_train,y_test))

        self.x_train = self.x_train.astype('float32') / 255


        # If subtract pixel mean is enabled
        self.x_train_mean = np.mean(self.x_train, axis=0)

        if subtract_pixel_mean:
            self.subtract_pixel_mean = 1
            self.x_train -= self.x_train_mean
        else:
            self.subtract_pixel_mean = 0

        self.x_train,self.y_train = shuffle(self.x_train,self.y_train)

        #Change Data format to channels_last
        img_rows = self.x_train.shape[1]
        img_cols = self.x_train.shape[2]
        channels = n_channels

        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, channels)

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels))

        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(64, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 14x14

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(128, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 7x7

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(256, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 512)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x/8 , size_y/8 , 512))

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(256, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(128, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(64, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(32, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        if(self.subtract_pixel_mean):
            recon = Conv2D(n_channels, (3, 3), activation='tanh', padding='same')
        else:
            recon = Conv2D(n_channels, (3, 3), activation='sigmoid', padding='same')

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



        adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if(loss=='mse'):
            self.loss_fn = mean_squared_error
        elif(loss=='dssim'):
            self.loss_fn = dssim
        elif(loss=='bce'):
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

        inp_assignments = Input(shape=(1,))

        self.y_obj = CustomClusterLayer_Test(self.loss_fn,self.means,self.lamda,self.n_clusters,h_units,self.assignment_lamda)

        y = self.y_obj([inp, ae18, encoder,inp_assignments])

        self.decoder = Model(inputs=[dinp], outputs=[de18])

        self.ae = Model(inputs=[inp,inp_assignments], outputs=[y])

        self.ae.compile(optimizer=adam_ae, loss=None)

        if(os.path.isfile(os.path.join(model_store,'weights.h5'))):
            print "LOADING MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store,'weights.h5'))


    def fit_model_ae(self, verbose=1, n_initial_epochs=10, n_train_epochs=10, earlystopping=False, patience=10, least_loss=1e-5):

        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        # start_initial epoch_of_training
        feats = self.encoder.predict(self.x_train)
        self.cluster_assigns = self.get_assigns(self.means,feats)

        self.ae.fit([self.x_train,self.cluster_assigns], shuffle=True, epochs=n_initial_epochs, batch_size=64, verbose=verbose)

        feats = self.encoder.predict(self.x_train)

        # Get means of predicted features
        km = KMeans(n_clusters=self.n_clusters, verbose=0, n_jobs=-1)
        km.fit(feats)

        self.means = np.copy(km.cluster_centers_).astype('float64')
        self.cluster_assigns = self.get_assigns(self.means, feats)
        print "$$$$$$$$$$$$$$$$$$"
        print "MEANS_INITIAL"
        print "$$$$$$$$$$$$$$$$$$"
        print self.means

        self.initial_means = np.copy(km.cluster_centers_)

        loss_track = 0
        lowest_loss_ever = 1000.0

        del km

        feats = self.encoder.predict(self.x_train)
        self.cluster_assigns = self.get_assigns(self.means,feats)

        # Make clustering loss weight to 1
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(1.0))
        K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))


        for i in range(0, n_train_epochs):

            print (i + 1), "/", n_train_epochs, ":"
            history = self.ae.fit([self.x_train,self.cluster_assigns], batch_size=self.batch_size, epochs=1, verbose=verbose, shuffle=True)
            current_loss = history.history['loss'][0]
            self.loss_list.append(history.history['loss'][0])

            feats = self.encoder.predict(self.x_train)
            self.cluster_assigns = self.get_assigns(self.means, feats)

            means_pre = np.copy(self.means)
            self.update_means(feats, self.cluster_assigns)

            #update cluster assigns for the next loop
            self.cluster_assigns = self.get_assigns(self.means, feats)

            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

            K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

            if (lowest_loss_ever - current_loss > least_loss):
                loss_track = 0
                print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                lowest_loss_ever = current_loss
            else:
                print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                loss_track += 1
                print "Loss track is :", loss_track, "/", patience

            if (earlystopping and loss_track > patience):
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "EARLY STOPPING AT EPOCH :", i
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                break

        self.ae.save_weights(filepath=os.path.join(self.model_store, 'weights.h5'))

    def get_assigns(self,means,feats):

        dist_matrix = cdist(feats,means)
        assigns = np.argmin(dist_matrix,axis=1)
        return assigns

    def update_means(self,feats,cluster_assigns):

        ck = np.zeros((self.means.shape[0]))
        for idx,i in enumerate(feats):
            ck[cluster_assigns[idx]]+=1
            cki = ck[cluster_assigns[idx]]
            self.means[cluster_assigns[idx]] -= (1/cki)*(self.means[cluster_assigns[idx]]-feats[idx])*self.clustering_lr

    def create_recon_images(self, num_recons):

        for l in range(0, num_recons):
            image_index = np.random.randint(0, len(self.x_train))
            xtm = self.x_train_mean * self.subtract_pixel_mean
            image_to_be_recon = np.uint8((self.x_train[image_index] + xtm) * 255)
            image_to_be_recon_shaped = np.expand_dims(self.x_train[image_index], axis=0)
            cl_assign = np.expand_dims(self.cluster_assigns[image_index], axis=0)
            class_of_image = self.y_train[image_index]

            reconstruction = np.uint8((self.ae.predict([image_to_be_recon_shaped,cl_assign])[0] + xtm) * 255)
            encoding = self.encoder.predict(image_to_be_recon_shaped)[0]

            n = encoding.shape[0]
            samples = np.ones((7, n)) * encoding + np.random.normal(0.0, 0.1, size=(7, n))


            predictions_samples = np.uint8((self.decoder.predict(samples) + xtm) * 255)

            f, axarr = plt.subplots(3, 3)
            plt.suptitle('Recon-class :' + str(class_of_image))
            axarr[0, 0].imshow(image_to_be_recon)
            axarr[0, 0].set_title('Original Image')

            axarr[0, 1].imshow(reconstruction)
            axarr[0, 1].set_title('R-Image')

            for i in range(0, 3):
                for j in range(0, 3):

                    if ((i == 0 and j == 0) or (i == 0 and j == 1)):
                        continue

                    axarr[i, j].imshow(predictions_samples[i * 3 + j - 2])
                    axarr[i, j].set_title('R-Sampling :' + str(i * 3 + j - 2))

            plt.savefig(os.path.join(self.model_store, 'recon-set' + str(l)))
            plt.close()

    def generate_loss_graph(self,graph_name):

        hfm, = plt.plot(self.loss_list,label='loss')
        plt.legend(handles=[hfm])
        plt.title('Losses per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot(self,graph_name):

        tsne_obj = TSNE(n_components=2,init='pca',random_state=0,verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        means = self.means
        full_array_feats = np.vstack((train_encodings,means))
        full_array_labels = np.vstack((self.y_train,np.ones((self.n_clusters,1))*10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"


        plt.scatter(x=train_tsne_embedding[:,0],y=train_tsne_embedding[:,1],c=colors,cmap=plt.get_cmap('Set3'),
                    alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot3d(self, graph_name):

        tsne_obj = TSNE(n_components=3, init='pca', random_state=0, verbose=0)
        train_encodings = self.encoder.predict(self.x_train)
        means = self.means
        full_array_feats = np.vstack((train_encodings, means))
        full_array_labels = np.vstack((self.y_train, np.ones((self.n_clusters, 1)) * 10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING 3D"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_tsne_embedding[:, 0], train_tsne_embedding[:, 1], train_tsne_embedding[:, 2], c=colors,
                   cmap=plt.get_cmap('Set3'),alpha=0.6)
        ax.set_title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        ax.legend()

        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def mean_displacement_distance(self):

        difference = self.means-self.initial_means

        print "####################################"
        print "Mean difference : "
        print  difference
        print "####################################"

        square = np.square(difference)
        sum_squares = np.sum(square,axis=1)

        print "####################################"
        print "Sum squares : "
        print  sum_squares
        print "####################################"


        displacement = np.sqrt(sum_squares)
        print "####################################"
        print "Mean displacement : ", displacement
        print "####################################"

    def generate_mean_displacement_graph(self,graph_name):

        mean_displacement = np.array(self.list_mean_disp)
        print mean_displacement.shape
        list_labels=[]

        for i in range(0,mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:,i],label='cluster_'+str(i+1))
            list_labels.append(x)

        plt.legend(handles=list_labels)
        plt.title('Mean displacements over iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Displacement Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def generate_assignment_graph(self,graph_name):

        feats = self.encoder.predict(self.x_train)
        assigns = self.get_assigns(self.means,feats)
        list_assigns = []

        for i in range(0,self.n_clusters):
            list_assigns.append(np.sum(assigns == i))

        plt.bar(x=range(0,self.n_clusters),height=list_assigns,width=0.25)
        plt.title('Cluster Assignments')
        plt.xlabel('Cluster ids')
        plt.ylabel('Number of assignments')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def decode_means(self,graph_name):

        xtm = self.x_train_mean * self.subtract_pixel_mean
        means_decoded = np.uint8((self.decoder.predict(self.means) + xtm) * 255)
        f, axarr = plt.subplots(2, 5)
        plt.suptitle('Means decoded')

        for i in range(0, 2):
            for j in range(0, 5):
                axarr[i, j].imshow(means_decoded[i*5+j])
                axarr[i, j].set_title('Mean :' + str(i*5+j))

        plt.savefig(os.path.join(self.model_store,graph_name))
        plt.close()

class Conv_LSTM_autoencoder:

    means = None
    initial_means = None
    list_mean_disp = []
    loss_list = []

    def __init__(self, model_store, size_y=32, size_x=32, n_channels=3, h_units=256, n_timesteps = 5,
                 loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-2, lr_model=1e-4, lamda=0.01, lamda_assign=0.01,n_gpus=1):

        self.clustering_lr = clustering_lr
        self.batch_size = batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model
        self.lamda = lamda
        self.means = np.zeros((n_clusters, h_units))
        self.cluster_assigns = None
        self.assignment_lamda = lamda_assign
        self.ntsteps = n_timesteps
        self.km = MiniBatchKMeans(n_clusters=self.n_clusters, verbose=0)


        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        # MODEL CREATION
        inp = Input(shape=(n_timesteps, size_y, size_x, n_channels))

        x1 = GaussianNoise(0.05)(inp)
        x1 = ConvLSTM2D(64, (3, 3), padding='same',return_sequences=True,dropout=0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x1)  # 14x14

        x1 = GaussianNoise(0.03)(x1)
        x1 = ConvLSTM2D(128, (3, 3), padding='same',return_sequences=True,dropout=0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x1)  # 7x7

        x1 = GaussianNoise(0.02)(x1)
        x1 = ConvLSTM2D(256, (3, 3), padding='same',return_sequences=True,dropout=0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 512 * n_timesteps)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((n_timesteps, size_x / 8, size_y / 8, 512))

        dec4 = TimeDistributed(UpSampling2D(size=(2, 2)))
        dec5 = ConvLSTM2D(256, (3, 3), padding='same',return_sequences=True)
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = TimeDistributed(UpSampling2D(size=(2, 2)))
        dec9 = ConvLSTM2D(128, (3, 3), padding='same',return_sequences=True)
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = TimeDistributed(UpSampling2D(size=(2, 2)))
        dec13 = ConvLSTM2D(64, (3, 3), padding='same',return_sequences=True)
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = ConvLSTM2D(32, (3, 3), padding='same',return_sequences=True)
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        recon = ConvLSTM2D(n_channels, (3, 3), activation='sigmoid', padding='same',return_sequences=True)

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

        adam_ae = Adam(lr=self.lr_model)
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

        inp_assignments = Input(shape=(1,))

        self.y_obj = CustomClusterLayer_Test(self.loss_fn, self.means, self.lamda, self.n_clusters, h_units,
                                             self.assignment_lamda)

        y = self.y_obj([inp, ae18, encoder, inp_assignments])

        self.decoder = Model(inputs=[dinp], outputs=[de18])

        self.ae = Model(inputs=[inp, inp_assignments], outputs=[y])

        if(n_gpus>1):
            self.ae = make_parallel(self.ae,n_gpus)

        self.ae.compile(optimizer=adam_ae, loss=None)

        if (os.path.isfile(os.path.join(model_store, 'ae_weights.h5'))):
            print "LOADING AE MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store, 'ae_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'decoder_weights.h5'))):
            print "LOADING DECODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.decoder.load_weights(os.path.join(model_store, 'decoder_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'encoder_weights.h5'))):
            print "LOADING ENCODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.encoder.load_weights(os.path.join(model_store, 'encoder_weights.h5'))


    def change_clustering_weight(self,flag):
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(flag))

    def set_graph_means(self,means):
        K.set_value(self.y_obj.means, K.cast_to_floatx(means))

    def kmeans_on_data(self,batch):

        feats = self.encoder.predict(batch)
        # Get means of predicted features
        self.km.partial_fit(feats)

        return True

    def kmeans_conclude(self):

        self.means = np.copy(self.km.cluster_centers_).astype('float64')
        print "$$$$$$$$$$$$$$$$$$"
        print "MEANS_INITIAL"
        print "$$$$$$$$$$$$$$$$$$"
        print self.means

        self.initial_means = np.copy(self.km.cluster_centers_)

        return True

    def fit_model_ae_on_batch(self, batch):

        # start_initial epoch_of_training
        feats = self.encoder.predict(batch)
        self.cluster_assigns = self.get_assigns(self.means, feats)

        current_loss = self.ae.train_on_batch([batch, self.cluster_assigns],None)

        self.loss_list.append(current_loss)

    def update_means_on_batch(self,batch):
        feats = self.encoder.predict(batch)
        self.cluster_assigns = self.get_assigns(self.means, feats)
        means_pre = np.copy(self.means)
        self.update_means(feats, self.cluster_assigns)

        self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))
        self.set_graph_means(self.means)

    def get_assigns(self, means, feats):

        dist_matrix = cdist(feats, means)
        assigns = np.argmin(dist_matrix, axis=1)
        return assigns

    def update_means(self, feats, cluster_assigns):

        ck = np.zeros((self.means.shape[0]))

        for idx, i in enumerate(feats):
            ck[cluster_assigns[idx]] += 1
            cki = ck[cluster_assigns[idx]]
            self.means[cluster_assigns[idx]] -= (1 / cki) * (self.means[cluster_assigns[idx]] - feats[idx]) * self.clustering_lr

    def generate_loss_graph(self, graph_name):

        hfm, = plt.plot(self.loss_list, label='loss')
        plt.legend(handles=[hfm])
        plt.title('Losses per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot(self, graph_name,batch):

        tsne_obj = TSNE(n_components=2, init='pca', random_state=0, verbose=0)
        train_encodings = self.encoder.predict(batch)
        means = self.means
        full_array_feats = np.vstack((train_encodings, means))
        cla = self.get_assigns(means,train_encodings)
        full_array_labels = np.vstack((cla, np.ones((self.n_clusters, 1)) * 10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        plt.scatter(x=train_tsne_embedding[:, 0], y=train_tsne_embedding[:, 1], c=colors, cmap=plt.get_cmap('Set3'),
                    alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot3d(self, graph_name,batch):

        tsne_obj = TSNE(n_components=3, init='pca', random_state=0, verbose=0)
        train_encodings = self.encoder.predict(batch)
        means = self.means
        full_array_feats = np.vstack((train_encodings, means))
        cla = self.get_assigns(means, train_encodings)
        full_array_labels = np.vstack((cla, np.ones((self.n_clusters, 1)) * 10))
        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING 3D"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_tsne_embedding[:, 0], train_tsne_embedding[:, 1], train_tsne_embedding[:, 2], c=colors,
                   cmap=plt.get_cmap('Set3'), alpha=0.6)
        ax.set_title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        ax.legend()

        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
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

        for i in range(0, mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:, i], label='cluster_' + str(i + 1))
            list_labels.append(x)

        plt.legend(handles=list_labels)
        plt.title('Mean displacements over iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Displacement Value')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def generate_assignment_graph(self, graph_name, batch):

        feats = self.encoder.predict(batch)
        assigns = self.get_assigns(self.means, feats)
        list_assigns = []

        for i in range(0, self.n_clusters):
            list_assigns.append(np.sum(assigns == i))

        plt.bar(x=range(0, self.n_clusters), height=list_assigns, width=0.25)
        plt.title('Cluster Assignments')
        plt.xlabel('Cluster ids')
        plt.ylabel('Number of assignments')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def do_gif_recon(self,input_cuboid,name):


        input_cuboid = np.expand_dims(input_cuboid,0)
        output_cuboid = self.ae.predict([input_cuboid,np.array([0])])

        input_cuboid = np.uint8(input_cuboid[0]*255.0)
        output_cuboid = np.uint8(output_cuboid[0]*255.0)



        for i in range(0,len(input_cuboid)):

            f, (ax1, ax2) = plt.subplots(2, 1)

            ax1.imshow(input_cuboid[i])
            ax1.set_title('Input Cuboid')
            ax1.set_axis_off()

            ax2.imshow(output_cuboid[i])
            ax2.set_title('Output Cuboid')
            ax2.set_axis_off()

            plt.axis('off')
            plt.savefig(os.path.join(self.model_store, str(i)+'.png'), bbox_inches='tight')
            plt.close()

        images = []
        for i in range(0, len(input_cuboid)):
            images.append(imageio.imread(os.path.join(self.model_store, str(i)+'.png')))

        imageio.mimsave(os.path.join(self.model_store, name+'.gif'), images)

        return True

    def save_weights(self):
        self.ae.save_weights(filepath=os.path.join(self.model_store, 'ae_weights.h5'))
        self.decoder.save_weights(filepath=os.path.join(self.model_store, 'decoder_weights.h5'))
        self.encoder.save_weights(filepath=os.path.join(self.model_store, 'encoder_weights.h5'))

class Conv_LSTM_autoencoder_nostream:

    means = None
    initial_means = None
    list_mean_disp = []
    loss_list = []


    def __init__(self, model_store, size_y=32, size_x=32, n_channels=3, h_units=256, n_timesteps=5,
                 loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-2, lr_model=1e-4, lamda=0.01,
                 lamda_assign=0.01, n_gpus=1,gs=False,notrain=False,reverse=False,data_folder='data_store'):

        self.clustering_lr = clustering_lr
        self.batch_size = batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model
        self.lamda = lamda
        self.means = np.zeros((n_clusters, h_units))
        self.cluster_assigns = None
        self.assignment_lamda = lamda_assign
        self.ntsteps = n_timesteps
        self.km = MiniBatchKMeans(n_clusters=self.n_clusters, verbose=0)
        self.x_train = [100,10,10]
        self.gs=gs
        self.notrain = notrain
        self.reverse = reverse

        self.dat_folder = data_folder

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        # MODEL CREATION
        inp = Input(shape=(n_timesteps, size_y, size_x, n_channels))

        x1 = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True, dropout=0.5)(inp)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x1)  # 14x14

        x1 = GaussianNoise(0.03)(x1)
        x1 = ConvLSTM2D(128, (3, 3), padding='same', return_sequences=True, dropout=0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x1)  # 7x7

        x1 = GaussianNoise(0.02)(x1)
        x1 = ConvLSTM2D(256, (3, 3), padding='same', return_sequences=True, dropout=0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 512 * n_timesteps)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((n_timesteps, size_x / 8, size_y / 8, 512))

        dec4 = TimeDistributed(UpSampling2D(size=(2, 2)))
        dec5 = ConvLSTM2D(256, (3, 3), padding='same', return_sequences=True)
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = TimeDistributed(UpSampling2D(size=(2, 2)))
        dec9 = ConvLSTM2D(128, (3, 3), padding='same', return_sequences=True)
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = TimeDistributed(UpSampling2D(size=(2, 2)))
        dec13 = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        recon = ConvLSTM2D(n_channels, (3, 3), activation='sigmoid', padding='same', return_sequences=True)

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

        adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if (loss == 'mse'):
            self.loss_fn = mean_squared_error
        elif (loss == 'dssim'):
            self.loss_fn = dssim
        elif (loss == 'bce'):
            self.loss_fn = binary_crossentropy

        self.encoder = Model(inputs=[inp], outputs=[encoder])

        if (n_gpus > 1):
            self.encoder = make_parallel(self.encoder, n_gpus)

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

        inp_assignments = Input(shape=(1,))

        self.y_obj = CustomClusterLayer_Test(self.loss_fn, self.means, self.lamda, self.n_clusters, h_units,
                                             self.assignment_lamda,self.reverse)

        y = self.y_obj([inp, ae18, encoder, inp_assignments])

        self.decoder = Model(inputs=[dinp], outputs=[de18])

        if (n_gpus > 1):
            self.decoder = make_parallel(self.decoder, n_gpus)

        self.ae = Model(inputs=[inp, inp_assignments], outputs=[y])

        if (n_gpus > 1):
            self.ae = make_parallel(self.ae, n_gpus)

        self.ae.compile(optimizer=adam_ae, loss=None)

        if (os.path.isfile(os.path.join(model_store, 'ae_weights.h5'))):
            print "LOADING AE MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store, 'ae_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'decoder_weights.h5'))):
            print "LOADING DECODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.decoder.load_weights(os.path.join(model_store, 'decoder_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'encoder_weights.h5'))):
            print "LOADING ENCODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.encoder.load_weights(os.path.join(model_store, 'encoder_weights.h5'))

        if(notrain):
            self.initial_means = np.load(os.path.join(self.model_store,'initial_means.npy'))
            self.means = np.load(os.path.join(self.model_store, 'means.npy'))

            with open(os.path.join(self.model_store,'losslist.pkl'), 'rb') as f:
                self.loss_list = pickle.load(f)
            with open(os.path.join(self.model_store,'meandisp.pkl'), 'rb') as f:
                self.list_mean_disp = pickle.load(f)

    def set_x_train(self,id):
        del(self.x_train)
        print "Loading Chapter : ", 'chapter_'+str(id)+'.npy'
        self.x_train = np.load(os.path.join(self.dat_folder,'chapter_'+str(id)+'.npy'))

    def fit_model_ae(self, verbose=1, n_initial_epochs=10, n_train_epochs=10, earlystopping=False, patience=10, least_loss=1e-5,n_chapters=20):

        if(self.notrain):
            return True
        
        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        # start_initial epoch_of_training

        for i in range(0,n_chapters):

            self.set_x_train(i)
            feats = self.encoder.predict(self.x_train)
            self.cluster_assigns = self.get_assigns(self.means, feats)
            self.ae.fit([self.x_train,self.cluster_assigns], shuffle=True, epochs=n_initial_epochs, batch_size=self.batch_size, verbose=verbose)

            del feats
            del self.cluster_assigns



        # Get means of predicted features
        for i in range(0, n_chapters):
            self.set_x_train(i)
            feats = self.encoder.predict(self.x_train)
            self.km.partial_fit(feats)
            del feats


        self.means = np.copy(self.km.cluster_centers_).astype('float64')

        print "$$$$$$$$$$$$$$$$$$"
        print "MEANS_INITIAL"
        print "$$$$$$$$$$$$$$$$$$"
        print self.means

        self.initial_means = np.copy(self.km.cluster_centers_)
        np.save(os.path.join(self.model_store, 'initial_means.npy'), self.initial_means)

        loss_track = 0
        lowest_loss_ever = 1000.0


        # Make clustering loss weight to 1
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(1.0))
        K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))


        for i in range(0, n_train_epochs):


            print (i + 1), "/", n_train_epochs, ":"

            for j in range(0, n_chapters):
                self.set_x_train(j)
                feats = self.encoder.predict(self.x_train)
                self.cluster_assigns = self.get_assigns(self.means, feats)
                history = self.ae.fit([self.x_train,self.cluster_assigns], batch_size=self.batch_size, epochs=1, verbose=verbose, shuffle=True)
                current_loss = history.history['loss'][0]
                self.loss_list.append(history.history['loss'][0])

                del feats
                feats = self.encoder.predict(self.x_train)

                del self.cluster_assigns
                self.cluster_assigns = self.get_assigns(self.means, feats)

                means_pre = np.copy(self.means)
                self.update_means(feats, self.cluster_assigns)

                #update cluster assigns for the next loop
                del self.cluster_assigns
                self.cluster_assigns = self.get_assigns(self.means, feats)

                self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))


                K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))


                if (lowest_loss_ever - current_loss > least_loss):
                    loss_track = 0
                    print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                    lowest_loss_ever = current_loss
                else:
                    print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                    loss_track += 1
                    print "Loss track is :", loss_track, "/", patience

                if (earlystopping and loss_track > patience):
                    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    print "EARLY STOPPING AT EPOCH :", i
                    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    break


            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            print "PICKLING LISTS AND SAVING WEIGHTS"
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

            with open(os.path.join(self.model_store,'losslist.pkl'), 'wb') as f:
                pickle.dump(self.loss_list, f)

            with open(os.path.join(self.model_store,'meandisp.pkl'),'wb') as f:
                pickle.dump(self.list_mean_disp,f)

            np.save(os.path.join(self.model_store, 'means.npy'), self.means)

            self.save_weights()

    def fit_model_ae_chaps(self, verbose=1, n_initial_chapters=10, earlystopping=False, patience=10,  least_loss=1e-5, n_chapters=20):

        if (self.notrain):
            return True

        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        # start_initial chapters training

        for i in range(0, n_initial_chapters):
            self.set_x_train(i)
            feats = self.encoder.predict(self.x_train)
            self.cluster_assigns = self.get_assigns(self.means, feats)
            history = self.ae.fit([self.x_train, self.cluster_assigns], shuffle=True, epochs=1,
                        batch_size=self.batch_size, verbose=verbose)
            self.loss_list.append(history.history['loss'][0])

            del feats
            del self.cluster_assigns

        # Get means of predicted features
        for i in range(0, n_initial_chapters):
            self.set_x_train(i)
            feats = self.encoder.predict(self.x_train)
            self.km.partial_fit(feats)
            del feats

        self.means = np.copy(self.km.cluster_centers_).astype('float64')

        print "$$$$$$$$$$$$$$$$$$"
        print "MEANS_INITIAL"
        print "$$$$$$$$$$$$$$$$$$"
        print self.means

        self.initial_means = np.copy(self.km.cluster_centers_)
        np.save(os.path.join(self.model_store, 'initial_means.npy'), self.initial_means)

        loss_track = 0
        lowest_loss_ever = 1000.0

        # Make clustering loss weight to 1
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(1.0))
        K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))



        for j in range(n_initial_chapters, n_chapters):

            print (j), "/", n_chapters, ":"
            self.set_x_train(j)
            feats = self.encoder.predict(self.x_train)
            self.cluster_assigns = self.get_assigns(self.means, feats)
            history = self.ae.fit([self.x_train, self.cluster_assigns], batch_size=self.batch_size, epochs=1,
                                  verbose=verbose, shuffle=True)
            current_loss = history.history['loss'][0]
            self.loss_list.append(history.history['loss'][0])

            del feats
            feats = self.encoder.predict(self.x_train)

            del self.cluster_assigns
            self.cluster_assigns = self.get_assigns(self.means, feats)

            means_pre = np.copy(self.means)
            self.update_means(feats, self.cluster_assigns)

            # update cluster assigns for the next loop
            del self.cluster_assigns
            self.cluster_assigns = self.get_assigns(self.means, feats)

            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

            K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

            if (lowest_loss_ever - current_loss > least_loss):
                loss_track = 0
                print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                lowest_loss_ever = current_loss
            else:
                print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                loss_track += 1
                print "Loss track is :", loss_track, "/", patience

            if (earlystopping and loss_track > patience):
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "EARLY STOPPING AT EPOCH :", i
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                break

            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            print "PICKLING LISTS AND SAVING WEIGHTS"
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

            with open(os.path.join(self.model_store, 'losslist.pkl'), 'wb') as f:
                pickle.dump(self.loss_list, f)

            with open(os.path.join(self.model_store, 'meandisp.pkl'), 'wb') as f:
                pickle.dump(self.list_mean_disp, f)

            np.save(os.path.join(self.model_store, 'means.npy'), self.means)

            self.save_weights()

        return True

    def get_assigns(self,means,feats):

        dist_matrix = cdist(feats,means)
        assigns = np.argmin(dist_matrix,axis=1)
        return assigns

    def update_means(self,feats,cluster_assigns):

        ck = np.zeros((self.means.shape[0]))
        for idx,i in enumerate(feats):
            ck[cluster_assigns[idx]]+=1
            cki = ck[cluster_assigns[idx]]
            self.means[cluster_assigns[idx]] -= (1/cki)*(self.means[cluster_assigns[idx]]-feats[idx])*self.clustering_lr

    def create_recons(self,n_recons):

        for i in range(0,n_recons):
            self.do_gif_recon(self.x_train[np.random.randint(0,len(self.x_train))],'recon_'+str(i))

        return True

    def do_gif_recon(self,input_cuboid,name):


        input_cuboid = np.expand_dims(input_cuboid,0)
        output_cuboid = self.ae.predict([input_cuboid,np.array([0])])

        input_cuboid = input_cuboid[0]
        output_cuboid = output_cuboid[0]

        for i in range(0,len(input_cuboid)):

            f, (ax1, ax2) = plt.subplots(2, 1)

            if(self.gs):
                ax1.imshow(input_cuboid[i].reshape(self.size_y,self.size_x))
            else:
                ax1.imshow(input_cuboid[i])
            ax1.set_title('Input Cuboid')
            ax1.set_axis_off()

            idx = (-i - 1) if self.reverse else i

            if(self.gs):

                ax2.imshow(output_cuboid[idx].reshape(self.size_y,self.size_x))
            else:
                ax2.imshow(output_cuboid[idx])
            ax2.set_title('Output Cuboid')
            ax2.set_axis_off()

            plt.axis('off')
            plt.savefig(os.path.join(self.model_store, str(i)+'.png'), bbox_inches='tight')
            plt.close()

        images = []
        for i in range(0, len(input_cuboid)):
            images.append(imageio.imread(os.path.join(self.model_store, str(i)+'.png')))

        imageio.mimsave(os.path.join(self.model_store, name+'.gif'), images)

        return True

    def save_gifs(self,input_cuboid,name):


        for i in range(0,len(input_cuboid)):

            f, (ax1) = plt.subplots(1, 1)

            idx = (-i - 1) if self.reverse else i

            if(self.gs):
                ax1.imshow(input_cuboid[idx].reshape(self.size_y,self.size_x))
            else:
                ax1.imshow(input_cuboid[idx])
            ax1.set_title('Input Cuboid')
            ax1.set_axis_off()

            plt.axis('off')
            plt.savefig(os.path.join(self.model_store, str(i)+'.png'), bbox_inches='tight')
            plt.close()

        images = []
        for i in range(0, len(input_cuboid)):
            images.append(imageio.imread(os.path.join(self.model_store, str(i)+'.png')))

        imageio.mimsave(os.path.join(self.model_store, name+'.gif'), images)

        return True

    def save_weights(self):
        self.ae.save_weights(filepath=os.path.join(self.model_store, 'ae_weights.h5'))
        self.decoder.save_weights(filepath=os.path.join(self.model_store, 'decoder_weights.h5'))
        self.encoder.save_weights(filepath=os.path.join(self.model_store, 'encoder_weights.h5'))

    def generate_loss_graph(self,graph_name):

        hfm, = plt.plot(self.loss_list,label='loss')
        plt.legend(handles=[hfm])
        plt.title('Losses per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def get_encodings_and_assigns(self,n_chapters,total_chaps_trained_on):

        encodings_full = []

        for i in range(0,n_chapters):
            idx = np.random.randint(0,total_chaps_trained_on)
            self.set_x_train(idx)
            train_encodings = self.encoder.predict(self.x_train)

            encodings_full.append(train_encodings.tolist())
            del train_encodings

        flat_list_encodings = [item for sublist in encodings_full for item in sublist]

        cluster_assigns = self.get_assigns(self.means,np.array(flat_list_encodings))

        return np.array(flat_list_encodings), cluster_assigns

    def create_tsne_plot(self,graph_name,n_chapters,total_chaps_trained):

        tsne_obj = TSNE(n_components=2,init='pca',random_state=0,verbose=0)

        train_encodings, cluster_assigns = self.get_encodings_and_assigns(n_chapters=n_chapters,
                                                                          total_chaps_trained_on=total_chaps_trained)

        full_array_feats = np.vstack((train_encodings,self.means))
        full_array_labels = np.vstack((cluster_assigns.reshape(len(cluster_assigns),1),np.ones((self.n_clusters,1))*10))

        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"


        plt.scatter(x=train_tsne_embedding[:,0],y=train_tsne_embedding[:,1],c=colors,cmap=plt.get_cmap('Set3'),
                    alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot3d(self, graph_name,n_chapters,total_chaps_trained):

        tsne_obj = TSNE(n_components=3, init='pca', random_state=0, verbose=0)

        train_encodings, cluster_assigns = self.get_encodings_and_assigns(n_chapters=n_chapters,total_chaps_trained_on=total_chaps_trained)

        full_array_feats = np.vstack((train_encodings, self.means))
        full_array_labels = np.vstack((cluster_assigns.reshape(len(cluster_assigns), 1), np.ones((self.n_clusters, 1)) * 10))


        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING 3D"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_tsne_embedding[:, 0], train_tsne_embedding[:, 1], train_tsne_embedding[:, 2], c=colors,
                   cmap=plt.get_cmap('Set3'),alpha=0.6)
        ax.set_title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        ax.legend()

        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def mean_displacement_distance(self):

        difference = self.means-self.initial_means

        print "####################################"
        print "Mean difference : "
        print  difference
        print "####################################"

        square = np.square(difference)
        sum_squares = np.sum(square,axis=1)

        print "####################################"
        print "Sum squares : "
        print  sum_squares
        print "####################################"


        displacement = np.sqrt(sum_squares)
        print "####################################"
        print "Mean displacement : ", displacement
        print "####################################"

    def generate_mean_displacement_graph(self,graph_name):

        mean_displacement = np.array(self.list_mean_disp)
        print mean_displacement.shape
        list_labels=[]

        for i in range(0,mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:,i],label='cluster_'+str(i+1))
            list_labels.append(x)

        plt.legend(handles=list_labels)
        plt.title('Mean displacements over iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Displacement Value')
        plt.savefig(os.path.join(self.model_store,graph_name), bbox_inches='tight')
        plt.close()

    def generate_assignment_graph(self,graph_name,n_chapters,total_chaps_trained):

        feats, assigns = self.get_encodings_and_assigns(n_chapters=n_chapters,total_chaps_trained_on=total_chaps_trained)
        list_assigns = []

        for i in range(0,self.n_clusters):
            list_assigns.append(np.sum(assigns == i))

        plt.bar(x=range(0,self.n_clusters),height=list_assigns,width=0.25)
        plt.title('Cluster Assignments')
        plt.xlabel('Cluster ids')
        plt.ylabel('Number of assignments')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def decode_means(self,graph_name):

        means_decoded = np.uint8(self.decoder.predict(self.means) * 255)

        for i in range(0,len(means_decoded)):
            self.save_gifs(means_decoded[i],graph_name+'_'+str(i+1))

class Conv_autoencoder_nostream:
    means = None
    initial_means = None
    list_mean_disp = []
    loss_list = []

    def __init__(self, model_store, size_y=32, size_x=32, n_channels=3, h_units=256, n_timesteps=5,
                 loss='mse', batch_size=64, n_clusters=10, clustering_lr=1e-2, lr_model=1e-4, lamda=0.01,
                 lamda_assign=0.01, n_gpus=1,gs=False,notrain=False,reverse=False,data_folder='data_store'):

        self.clustering_lr = clustering_lr
        self.batch_size = batch_size
        self.model_store = model_store
        self.loss = loss
        self.n_clusters = n_clusters
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model
        self.lamda = lamda
        self.means = np.zeros((n_clusters, h_units))
        self.cluster_assigns = None
        self.assignment_lamda = lamda_assign
        self.ntsteps = n_timesteps
        self.km = MiniBatchKMeans(n_clusters=self.n_clusters, verbose=0)
        self.x_train = [100,10,10]
        self.gs=gs
        self.notrain = notrain
        self.reverse = reverse
        self.n_channels = n_channels

        self.dat_folder = data_folder

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels*n_timesteps))
        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(64, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 16x16

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(128, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 8x8

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(128, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 128)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x / 8, size_y / 8, 128))

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(256, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(128, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(64, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(32, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        recon = Conv2D(n_timesteps, (3, 3), activation='sigmoid', padding='same')

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

        adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if (loss == 'mse'):
            self.loss_fn = mean_squared_error
        elif (loss == 'dssim'):
            self.loss_fn = dssim
        elif (loss == 'bce'):
            self.loss_fn = binary_crossentropy

        self.encoder = Model(inputs=[inp], outputs=[encoder])

        if (n_gpus > 1):
            self.encoder = make_parallel(self.encoder, n_gpus)

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

        inp_assignments = Input(shape=(1,))

        self.y_obj = CustomClusterLayer_Test(self.loss_fn, self.means, self.lamda, self.n_clusters, h_units,
                                             self.assignment_lamda,self.reverse)

        y = self.y_obj([inp, ae18, encoder, inp_assignments])

        self.decoder = Model(inputs=[dinp], outputs=[de18])

        if (n_gpus > 1):
            self.decoder = make_parallel(self.decoder, n_gpus)

        self.ae = Model(inputs=[inp, inp_assignments], outputs=[y])

        if (n_gpus > 1):
            self.ae = make_parallel(self.ae, n_gpus)

        self.ae.compile(optimizer=adam_ae, loss=None)

        if (os.path.isfile(os.path.join(model_store, 'ae_weights.h5'))):
            print "LOADING AE MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store, 'ae_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'decoder_weights.h5'))):
            print "LOADING DECODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.decoder.load_weights(os.path.join(model_store, 'decoder_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'encoder_weights.h5'))):
            print "LOADING ENCODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.encoder.load_weights(os.path.join(model_store, 'encoder_weights.h5'))

        if(notrain):
            self.initial_means = np.load(os.path.join(self.model_store,'initial_means.npy'))
            self.means = np.load(os.path.join(self.model_store, 'means.npy'))

            with open(os.path.join(self.model_store,'losslist.pkl'), 'rb') as f:
                self.loss_list = pickle.load(f)
            with open(os.path.join(self.model_store,'meandisp.pkl'), 'rb') as f:
                self.list_mean_disp = pickle.load(f)

    def set_x_train(self, id):
        del (self.x_train)
        print "Loading Chapter : ", 'chapter_' + str(id) + '.npy'
        self.x_train = np.load(os.path.join(self.dat_folder, 'chapter_' + str(id) + '.npy'))

    def fit_model_ae_chaps(self, verbose=1, n_initial_chapters=10, earlystopping=False, patience=10, least_loss=1e-5,
                           n_chapters=20):

        if (self.notrain):
            return True

        # Make clustering loss weight to 0
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(0.0))

        # start_initial chapters training

        for i in range(0, n_initial_chapters):
            self.set_x_train(i)
            feats = self.encoder.predict(self.x_train)
            self.cluster_assigns = self.get_assigns(self.means, feats)
            history = self.ae.fit([self.x_train, self.cluster_assigns], shuffle=True, epochs=1,
                        batch_size=self.batch_size, verbose=verbose)
            self.loss_list.append(history.history['loss'][0])

            del feats
            del self.cluster_assigns

        return True

        # Get means of predicted features
        for i in range(0, n_initial_chapters):
            self.set_x_train(i)
            feats = self.encoder.predict(self.x_train)
            self.km.partial_fit(feats)
            del feats

        self.means = np.copy(self.km.cluster_centers_).astype('float64')

        print "$$$$$$$$$$$$$$$$$$"
        print "MEANS_INITIAL"
        print "$$$$$$$$$$$$$$$$$$"
        print self.means

        self.initial_means = np.copy(self.km.cluster_centers_)
        np.save(os.path.join(self.model_store, 'initial_means.npy'), self.initial_means)

        loss_track = 0
        lowest_loss_ever = 1000.0

        # Make clustering loss weight to 1
        K.set_value(self.y_obj.cl_loss_wt, K.cast_to_floatx(1.0))
        K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

        for j in range(n_initial_chapters, n_chapters):

            print (j), "/", n_chapters, ":"
            self.set_x_train(j)
            feats = self.encoder.predict(self.x_train)
            self.cluster_assigns = self.get_assigns(self.means, feats)
            history = self.ae.fit([self.x_train, self.cluster_assigns], batch_size=self.batch_size, epochs=1,
                                  verbose=verbose, shuffle=True)
            current_loss = history.history['loss'][0]
            self.loss_list.append(history.history['loss'][0])

            del feats
            feats = self.encoder.predict(self.x_train)

            del self.cluster_assigns
            self.cluster_assigns = self.get_assigns(self.means, feats)

            means_pre = np.copy(self.means)
            self.update_means(feats, self.cluster_assigns)

            # update cluster assigns for the next loop
            del self.cluster_assigns
            self.cluster_assigns = self.get_assigns(self.means, feats)

            self.list_mean_disp.append(np.sqrt(np.sum(np.square(self.means - means_pre), axis=1)))

            K.set_value(self.y_obj.means, K.cast_to_floatx(self.means))

            if (lowest_loss_ever - current_loss > least_loss):
                loss_track = 0
                print "LOSS IMPROVED FROM :", lowest_loss_ever, " to ", current_loss
                lowest_loss_ever = current_loss
            else:
                print "LOSS DEGRADED FROM :", lowest_loss_ever, " to ", current_loss
                loss_track += 1
                print "Loss track is :", loss_track, "/", patience

            if (earlystopping and loss_track > patience):
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print "EARLY STOPPING AT EPOCH :", i
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                break

            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            print "PICKLING LISTS AND SAVING WEIGHTS"
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

            with open(os.path.join(self.model_store, 'losslist.pkl'), 'wb') as f:
                pickle.dump(self.loss_list, f)

            with open(os.path.join(self.model_store, 'meandisp.pkl'), 'wb') as f:
                pickle.dump(self.list_mean_disp, f)

            np.save(os.path.join(self.model_store, 'means.npy'), self.means)

            self.save_weights()

        return True

    def get_assigns(self, means, feats):

        dist_matrix = cdist(feats, means)
        assigns = np.argmin(dist_matrix, axis=1)
        return assigns

    def update_means(self, feats, cluster_assigns):

        ck = np.zeros((self.means.shape[0]))
        for idx, i in enumerate(feats):
            ck[cluster_assigns[idx]] += 1
            cki = ck[cluster_assigns[idx]]
            self.means[cluster_assigns[idx]] -= (1 / cki) * (
            self.means[cluster_assigns[idx]] - feats[idx]) * self.clustering_lr

    def create_recons(self, n_recons):

        for i in range(0, n_recons):
            self.do_gif_recon(self.x_train[np.random.randint(0, len(self.x_train))], 'recon_' + str(i))

        return True

    def do_gif_recon(self, input_cuboid, name):

        input_cuboid = np.expand_dims(input_cuboid, 0)
        output_cuboid = self.ae.predict([input_cuboid, np.array([0])])

        input_cuboid = input_cuboid[0]
        output_cuboid = output_cuboid[0]

        for i in range(0, input_cuboid.shape[-1]/self.n_channels):

            f, (ax1, ax2) = plt.subplots(2, 1)

            if (self.gs):
                ax1.imshow(input_cuboid[:,:,i].reshape(self.size_y, self.size_x))
            else:
                ax1.imshow(input_cuboid[:,:,i*self.n_channels:(i+1)*self.n_channels])

            ax1.set_title('Input Cuboid')
            ax1.set_axis_off()

            idx = (-i - 1) if self.reverse else i

            if (self.gs):
                ax2.imshow(output_cuboid[:,:,idx].reshape(self.size_y, self.size_x))
            else:
                ax2.imshow(output_cuboid[:,:,idx*self.n_channels:(idx+1)*self.n_channels])
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

    def save_gifs(self, input_cuboid, name):

        for i in range(0, input_cuboid.shape[-1]/self.n_channels):

            f, (ax1) = plt.subplots(1, 1)

            idx = (-i - 1) if self.reverse else i

            if (self.gs):
                ax1.imshow(input_cuboid[:, :, i].reshape(self.size_y, self.size_x))
            else:
                ax1.imshow(input_cuboid[:, :, i * self.n_channels:(i + 1) * self.n_channels])

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
        self.ae.save_weights(filepath=os.path.join(self.model_store, 'ae_weights.h5'))
        self.decoder.save_weights(filepath=os.path.join(self.model_store, 'decoder_weights.h5'))
        self.encoder.save_weights(filepath=os.path.join(self.model_store, 'encoder_weights.h5'))

    def generate_loss_graph(self, graph_name):

        hfm, = plt.plot(self.loss_list, label='loss')
        plt.legend(handles=[hfm])
        plt.title('Losses per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

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

    def create_tsne_plot(self, graph_name, n_chapters, total_chaps_trained):

        tsne_obj = TSNE(n_components=2, init='pca', random_state=0, verbose=0)

        train_encodings, cluster_assigns = self.get_encodings_and_assigns(n_chapters=n_chapters,
                                                                          total_chaps_trained_on=total_chaps_trained)

        full_array_feats = np.vstack((train_encodings, self.means))
        full_array_labels = np.vstack(
            (cluster_assigns.reshape(len(cluster_assigns), 1), np.ones((self.n_clusters, 1)) * 10))

        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        plt.scatter(x=train_tsne_embedding[:, 0], y=train_tsne_embedding[:, 1], c=colors, cmap=plt.get_cmap('Set3'),
                    alpha=0.6)
        plt.colorbar()
        plt.title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def create_tsne_plot3d(self, graph_name, n_chapters, total_chaps_trained):

        tsne_obj = TSNE(n_components=3, init='pca', random_state=0, verbose=0)

        train_encodings, cluster_assigns = self.get_encodings_and_assigns(n_chapters=n_chapters,
                                                                          total_chaps_trained_on=total_chaps_trained)

        full_array_feats = np.vstack((train_encodings, self.means))
        full_array_labels = np.vstack(
            (cluster_assigns.reshape(len(cluster_assigns), 1), np.ones((self.n_clusters, 1)) * 10))

        colors = full_array_labels.reshape((full_array_labels.shape[0],))

        print "##############################"
        print "START TSNE FITTING 3D"
        print "##############################"
        train_tsne_embedding = tsne_obj.fit_transform(full_array_feats)

        print "##############################"
        print "TSNE FITTING DONE"
        print "##############################"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_tsne_embedding[:, 0], train_tsne_embedding[:, 1], train_tsne_embedding[:, 2], c=colors,
                   cmap=plt.get_cmap('Set3'), alpha=0.6)
        ax.set_title('TSNE EMBEDDINGS OF THE CLUSTER MEANS AND THE ENCODED FEATURES')
        ax.legend()

        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
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

        for i in range(0, mean_displacement.shape[1]):
            x, = plt.plot(mean_displacement[:, i], label='cluster_' + str(i + 1))
            list_labels.append(x)

        plt.legend(handles=list_labels)
        plt.title('Mean displacements over iterations')
        plt.xlabel('Iteration Index')
        plt.ylabel('Displacement Value')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def generate_assignment_graph(self, graph_name, n_chapters, total_chaps_trained):

        feats, assigns = self.get_encodings_and_assigns(n_chapters=n_chapters,
                                                        total_chaps_trained_on=total_chaps_trained)
        list_assigns = []

        for i in range(0, self.n_clusters):
            list_assigns.append(np.sum(assigns == i))

        plt.bar(x=range(0, self.n_clusters), height=list_assigns, width=0.25)
        plt.title('Cluster Assignments')
        plt.xlabel('Cluster ids')
        plt.ylabel('Number of assignments')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

    def decode_means(self, graph_name):

        means_decoded = np.uint8(self.decoder.predict(self.means) * 255)

        for i in range(0, len(means_decoded)):
            self.save_gifs(means_decoded[i], graph_name + '_' + str(i + 1))

class Conv_autoencoder_nostream_nocl:
    means = None
    initial_means = None
    list_mean_disp = []
    loss_list = []

    def __init__(self, model_store, size_y=32, size_x=32, n_channels=3, h_units=256, n_timesteps=5,
                 loss='mse', batch_size=64, n_clusters=10, lr_model=1e-4, lamda=0.01,
                 n_gpus=1,gs=False,notrain=False,reverse=False,data_folder='data_store'):

        self.batch_size = batch_size
        self.model_store = model_store
        self.loss = loss
        self.size_y = size_y
        self.size_x = size_x
        self.lr_model = lr_model
        self.lamda = lamda
        self.means = np.zeros((n_clusters, h_units))
        self.ntsteps = n_timesteps
        self.x_train = [100,10,10]
        self.gs=gs
        self.notrain = notrain
        self.reverse = reverse
        self.n_channels = n_channels

        self.dat_folder = data_folder

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)

        # MODEL CREATION
        inp = Input(shape=(size_y, size_x, n_channels*n_timesteps))
        x1 = GaussianNoise(0.05)(inp)
        x1 = Conv2D(128, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 16x16

        x1 = GaussianNoise(0.03)(x1)
        x1 = Conv2D(256, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 8x8

        x1 = GaussianNoise(0.02)(x1)
        x1 = Conv2D(512, (3, 3), padding='same')(x1)
        x1 = SpatialDropout2D(0.5)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)  # 4x4

        x1 = Flatten()(x1)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(units=h_units)(x1)
        x1 = LeakyReLU(alpha=0.2)(x1)
        encoder = BatchNormalization()(x1)

        dec1 = Dense(units=(size_y / 8) * (size_x / 8) * 512)
        dec2 = LeakyReLU(alpha=0.2)
        dec3 = Reshape((size_x / 8, size_y / 8, 512))

        dec4 = UpSampling2D(size=(2, 2))
        dec5 = Conv2D(256, (3, 3), padding='same')
        dec6 = LeakyReLU(alpha=0.2)
        dec7 = BatchNormalization()  # 8x8

        dec8 = UpSampling2D(size=(2, 2))
        dec9 = Conv2D(128, (3, 3), padding='same')
        dec10 = LeakyReLU(alpha=0.2)
        dec11 = BatchNormalization()  # 16x16

        dec12 = UpSampling2D(size=(2, 2))
        dec13 = Conv2D(64, (3, 3), padding='same')
        dec14 = LeakyReLU(alpha=0.2)
        dec15 = BatchNormalization()  # 32x32

        dec16 = Conv2D(32, (3, 3), padding='same')
        dec17 = LeakyReLU(alpha=0.2)  # 32x32

        recon = Conv2D(n_timesteps, (3, 3), activation='sigmoid', padding='same')

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

        adam_ae = Adam(lr=self.lr_model)
        dssim = DSSIMObjective(kernel_size=4)

        if (loss == 'mse'):
            self.loss_fn = mean_squared_error
        elif (loss == 'dssim'):
            self.loss_fn = dssim
        elif (loss == 'bce'):
            self.loss_fn = binary_crossentropy

        self.encoder = Model(inputs=[inp], outputs=[encoder])

        if (n_gpus > 1):
            self.encoder = make_parallel(self.encoder, n_gpus)

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

        inp_assignments = Input(shape=(1,))

        self.decoder = Model(inputs=[dinp], outputs=[de18])

        if (n_gpus > 1):
            self.decoder = make_parallel(self.decoder, n_gpus)

        self.ae = Model(inputs=[inp], outputs=[ae18])

        if (n_gpus > 1):
            self.ae = make_parallel(self.ae, n_gpus)

        self.ae.compile(optimizer=adam_ae, loss=self.loss_fn)

        if (os.path.isfile(os.path.join(model_store, 'ae_weights.h5'))):
            print "LOADING AE MODEL WEIGHTS FROM WEIGHTS FILE"
            self.ae.load_weights(os.path.join(model_store, 'ae_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'decoder_weights.h5'))):
            print "LOADING DECODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.decoder.load_weights(os.path.join(model_store, 'decoder_weights.h5'))

        if (os.path.isfile(os.path.join(model_store, 'encoder_weights.h5'))):
            print "LOADING ENCODER MODEL WEIGHTS FROM WEIGHTS FILE"
            self.encoder.load_weights(os.path.join(model_store, 'encoder_weights.h5'))

        if(notrain):
            self.initial_means = np.load(os.path.join(self.model_store,'initial_means.npy'))
            self.means = np.load(os.path.join(self.model_store, 'means.npy'))

            with open(os.path.join(self.model_store,'losslist.pkl'), 'rb') as f:
                self.loss_list = pickle.load(f)
            with open(os.path.join(self.model_store,'meandisp.pkl'), 'rb') as f:
                self.list_mean_disp = pickle.load(f)

    def set_x_train(self, id):
        del (self.x_train)
        print "Loading Chapter : ", 'chapter_' + str(id) + '.npy'
        self.x_train = np.load(os.path.join(self.dat_folder, 'chapter_' + str(id) + '.npy'))

    def fit_model_ae_chaps(self, verbose=1, n_initial_chapters=10):

        if (self.notrain):
            return True

        # start_initial chapters training
        for i in range(0, n_initial_chapters):
            self.set_x_train(i)
            feats = self.encoder.predict(self.x_train)
            history = self.ae.fit(x=self.x_train,y=self.x_train, shuffle=True, epochs=1,
                                  batch_size=self.batch_size, verbose=verbose)
            self.loss_list.append(history.history['loss'][0])

            del feats


        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        print "PICKLING LISTS AND SAVING WEIGHTS"
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

        with open(os.path.join(self.model_store, 'losslist.pkl'), 'wb') as f:
            pickle.dump(self.loss_list, f)

        with open(os.path.join(self.model_store, 'meandisp.pkl'), 'wb') as f:
            pickle.dump(self.list_mean_disp, f)

        np.save(os.path.join(self.model_store, 'means.npy'), self.means)

        self.save_weights()

        return True

    def create_recons(self, n_recons):

        for i in range(0, n_recons):
            self.do_gif_recon(self.x_train[np.random.randint(0, len(self.x_train))], 'recon_' + str(i))

        return True

    def do_gif_recon(self, input_cuboid, name):

        input_cuboid = np.expand_dims(input_cuboid, 0)
        output_cuboid = self.ae.predict(input_cuboid)

        input_cuboid = input_cuboid[0]
        output_cuboid = output_cuboid[0]

        for i in range(0, input_cuboid.shape[-1]/self.n_channels):

            f, (ax1, ax2) = plt.subplots(2, 1)

            if (self.gs):
                ax1.imshow(input_cuboid[:,:,i].reshape(self.size_y, self.size_x))
            else:
                ax1.imshow(input_cuboid[:,:,i*self.n_channels:(i+1)*self.n_channels])

            ax1.set_title('Input Cuboid')
            ax1.set_axis_off()

            idx = (-i - 1) if self.reverse else i

            if (self.gs):
                ax2.imshow(output_cuboid[:,:,idx].reshape(self.size_y, self.size_x))
            else:
                ax2.imshow(output_cuboid[:,:,idx*self.n_channels:(idx+1)*self.n_channels])
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

    def save_gifs(self, input_cuboid, name):

        for i in range(0, input_cuboid.shape[-1]/self.n_channels):

            f, (ax1) = plt.subplots(1, 1)

            idx = (-i - 1) if self.reverse else i

            if (self.gs):
                ax1.imshow(input_cuboid[:, :, i].reshape(self.size_y, self.size_x))
            else:
                ax1.imshow(input_cuboid[:, :, i * self.n_channels:(i + 1) * self.n_channels])

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
        self.ae.save_weights(filepath=os.path.join(self.model_store, 'ae_weights.h5'))
        self.decoder.save_weights(filepath=os.path.join(self.model_store, 'decoder_weights.h5'))
        self.encoder.save_weights(filepath=os.path.join(self.model_store, 'encoder_weights.h5'))

    def generate_loss_graph(self, graph_name):

        hfm, = plt.plot(self.loss_list, label='loss')
        plt.legend(handles=[hfm])
        plt.title('Losses per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()