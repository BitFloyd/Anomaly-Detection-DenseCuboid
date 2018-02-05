import numpy as np
import os
from skimage.io import imread
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import color,img_as_float
from collections import deque
import copy


class Cuboid:

    def __init__(self,cuboid,pixmap,surroundings,anomaly_status):

        self.cuboid=cuboid
        #Normalized cuboid data

        self.pixmap = pixmap
        #pixel coordinates of the cuboid in the frame

        self.surroundings = surroundings
        #surroundings = list of surrounding cuboids in order [rowmajor-past,rowmajor present top, left, right,
        #                                                     rowmajor bottom, rowmajor future] = 9+8+9 = 26 cuboids

        self.anomaly_status = anomaly_status
        #True if the cuboid has an anomaly in it.


class Video_Stream_UCSD:

    video_path = 'INIT_PATH_TO_UCSD'
    video_train_test = 'Train'
    size_y = 8
    size_x = 8
    timesteps = 4
    frame_size = (128, 128)
    data_max_possible = 255.0
    seek = -1
    list_images_relevant_full_dset = None
    list_images_relevant_gt_full_dset = None
    seek_dict = {}
    seek_dict_gt = {}

    list_cuboids = []
    list_cuboids_pixmap = []
    list_cuboids_anomaly = []
    list_all_cuboids = []
    list_all_cuboids_gt = []

    def __init__(self, video_path,video_train_test, size_y,size_x,timesteps,num=-1,ts_first_or_last='first',strides=1):

        # Initialize-video-params
        self.video_path = video_path
        self.video_train_test = video_train_test
        self.size_y = size_y
        self.size_x = size_x
        self.timesteps = timesteps
        self.ts_first_or_last = ts_first_or_last
        self.list_images_relevant_full_dset, self.list_images_relevant_gt_full_dset = make_file_list(video_path, train_test=video_train_test,n_frames=timesteps,num=num)
        self.strides = strides

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

    def get_next_cuboids_from_stream(self):
        self.seek+=1

        if(self.seek>=len(self.seek_dict)):
            self.seek=-1
            return np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0])
        else:
            self.list_cuboids, self.list_cuboids_pixmap, self.list_cuboids_anomaly, self.list_all_cuboids, self.list_all_cuboids_gt = \
            make_cuboids_for_stream(self,self.seek_dict[self.seek], self.seek_dict_gt[self.seek], self.size_x, self.size_y,
                                    test_or_train=self.video_train_test, ts_first_last = self.ts_first_or_last,strides=self.strides)

        return self.list_cuboids, self.list_cuboids_pixmap, self.list_cuboids_anomaly, np.array(self.list_all_cuboids), np.array(self.list_all_cuboids_gt)

    def reset_stream(self):
        self.seek = -1

class Video_Stream_ARTIF:

    video_path = 'INIT_PATH_TO_UCSD'
    video_train_test = 'Test'
    size_y = 8
    size_x = 8
    timesteps = 4
    frame_size = (128, 128)
    data_max_possible = 255.0
    seek = -1
    list_images_relevant_full_dset = None
    list_images_relevant_gt_full_dset = None
    seek_dict = {}
    seek_dict_gt = {}

    list_cuboids = []
    list_cuboids_pixmap = []
    list_cuboids_anomaly = []
    list_all_cuboids = []
    list_all_cuboids_gt = []

    def __init__(self, video_path,video_train_test, size_y,size_x,timesteps,num=-1,ts_first_or_last='first',strides=1,tstrides=1):

        # Initialize-video-params
        self.video_path = video_path
        self.video_train_test = video_train_test
        self.size_y = size_y
        self.size_x = size_x
        self.timesteps = timesteps
        self.ts_first_or_last = ts_first_or_last
        self.list_images_relevant_full_dset, self.list_images_relevant_gt_full_dset = make_file_list(video_path, train_test=video_train_test,n_frames=timesteps,num=num,tstrides=tstrides)
        self.strides = strides

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

            list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly, list_all_cuboids, list_all_cuboids_gt = \
            make_cuboids_for_stream(self,self.seek_dict[self.seek], self.seek_dict_gt[self.seek], self.size_x, self.size_y,
                                    test_or_train=self.video_train_test, ts_first_last = self.ts_first_or_last,strides=self.strides,gs=gs)

            del (self.list_cuboids[:])
            self.list_cuboids = copy.copy(list_cuboids)
            del(self.list_cuboids_pixmap[:])
            self.list_cuboids_pixmap = copy.copy(list_cuboids_pixmap)
            del(self.list_cuboids_anomaly[:])
            self.list_cuboids_anomaly = copy.copy(list_cuboids_anomaly)
            del(self.list_all_cuboids[:])
            self.list_all_cuboids = copy.copy(list_all_cuboids)
            del(self.list_all_cuboids_gt[:])
            self.list_all_cuboids_gt = copy.copy(list_all_cuboids_gt)

        return list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly, np.array(list_all_cuboids), np.array(list_all_cuboids_gt)

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


def make_cuboids_for_stream(stream,list_images,list_images_gt,size_x,size_y,test_or_train='Train', ts_first_last = 'first',strides=1,gs=True):


    list_cuboids = deque(stream.list_cuboids)
    list_cuboids_pixmap = deque(stream.list_cuboids_pixmap)
    list_cuboids_anomaly = deque(stream.list_cuboids_anomaly)
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

        local_collection = imread_collection(list_images[i],as_grey=gs)

        n_frames = len(local_collection)
        frame_size = local_collection[0].shape

        if(test_or_train=='Test'):
            local_collection_gt = imread_collection(list_images_gt[i],as_grey=gs)


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
                lcgt = np.zeros((n_frames, frame_size_y, frame_size_x, n_channels))

            for l in range(0, len(local_collection)):
                lc[l] = local_collection[l].reshape(frame_size_y, frame_size_x, n_channels)
                if (test_or_train == 'Test'):
                    lcgt[l] = local_collection_gt[l].reshape(frame_size_y, frame_size_x, n_channels)

            del local_collection
            local_collection = lc

            if (test_or_train == 'Test'):
                del local_collection_gt
                local_collection_gt = lcgt

        elif (ts_first_last == 'last'):

            lc = np.zeros((frame_size_y, frame_size_x, n_channels*n_frames))

            if (test_or_train == 'Test'):
                lcgt = np.zeros((frame_size_y, frame_size_x, n_channels*n_frames))

            for l in range(0,len(local_collection)):
                lc[:,:,l*n_channels:(l+1)*n_channels] = local_collection[l].reshape(frame_size_y, frame_size_x, n_channels)


            if (test_or_train == 'Test'):
                for l in range(0, len(local_collection_gt)):
                    lcgt[:, :, l * n_channels:(l + 1) * n_channels] = local_collection_gt[l].reshape(frame_size_y,frame_size_x, n_channels)


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
                if(test_or_train=='Test'):

                    if (ts_first_last == 'first'):
                        anomaly_gt_sum = np.sum(local_collection_gt[:, start_rows:end_rows, start_cols:end_cols, :])
                    elif (ts_first_last == 'last'):
                        anomaly_gt_sum = np.sum(local_collection_gt[start_rows:end_rows, start_cols:end_cols, :])

                    if(anomaly_gt_sum>0):
                        anomaly_gt=True


                list_cuboids_local.append(cuboid_data)

                if(start):
                    list_all_cuboids.append(cuboid_data)
                else:
                    list_all_cuboids.popleft()
                    list_all_cuboids.append(cuboid_data)

                list_cuboids_pixmap_local.append((j,k))
                list_cuboids_anomaly_local.append(anomaly_gt)

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

        if(start):
            list_cuboids.append(array_cuboids)
            list_cuboids_pixmap.append(pixmap_cuboids)
            list_cuboids_anomaly.append(anomaly_gt_cuboids)
        else:
            list_cuboids.popleft()
            list_cuboids.append(array_cuboids)
            list_cuboids_pixmap.popleft()
            list_cuboids_pixmap.append(pixmap_cuboids)
            list_cuboids_anomaly.popleft()
            list_cuboids_anomaly.append(anomaly_gt_cuboids)

    return list(list_cuboids), list(list_cuboids_pixmap), list(list_cuboids_anomaly), list(list_all_cuboids), list(list_all_cuboids_gt)

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


