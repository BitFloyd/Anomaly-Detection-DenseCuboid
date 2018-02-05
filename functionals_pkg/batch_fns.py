import numpy as np
from sklearn.utils import shuffle
from data_pkg.data_fns import  Cuboid

def norm_batch(batch,min=0,max=255.0,sub_mean=False):

    if(sub_mean):
        mid = (max-min)/2.0
        batch = (batch-mid)/mid
    else:
        batch = batch/max

    return batch

def get_mini_batches(batch,batch_size=32):

    batch = shuffle(batch)

    list_mini_batches = []

    n_mini_batches = int(len(batch)/batch_size)

    for i in range(0,n_mini_batches):
        list_mini_batches.append(batch[i*batch_size:(i+1)*batch_size])

    if(n_mini_batches*batch_size<len(batch)):
        list_mini_batches.append(batch[n_mini_batches*batch_size:])

    return list_mini_batches

def train_on_batch(model,batch,batch_size,train=True):

    lmb = get_mini_batches(batch,batch_size)
    for minib in lmb:
        model.fit_model_ae_on_batch(minib)
        if(train):
            model.update_means_on_batch(minib)

    return True

def kmeans_on_batch(model,batch,batch_size):
    lmb = get_mini_batches(batch,batch_size)
    for minib in lmb:
        model.kmeans_on_data(minib)

    return True

def detect_motion(cuboid,ts_pos=0,gs=0):

    motion_sum = 0
    list_frames = []
    list_diff = []

    if(ts_pos==0):
        #lstm regime
        for i in range(0,cuboid.shape[ts_pos]):
            list_frames.append(cuboid[i])

    else:
        #conv regime
        if(gs==1):
            for i in range(0,cuboid.shape[ts_pos]):
                list_frames.append(np.expand_dims(cuboid[:,:,i],axis=-1))

        else:
            for i in range(0,cuboid.shape[ts_pos],3):
                list_frames.append(cuboid[:,:,i:i+3])

    #list_frames contains a list of frames. list_frames should have n_timesteps frames in it.

    for i in range(0,len(list_frames)-1):
        list_diff.append(np.mean(np.any(np.abs(list_frames[i]-list_frames[i+1]),axis=2)))

    return np.mean(list_diff)

def get_variances(cuboids_batch,ts_pos=0,gs=0):

    list_mot_hist=[]
    for j in cuboids_batch:
        # sum_variances = np.mean(np.var(j,axis=ts_pos))
        sum_variances = detect_motion(j,ts_pos,gs)
        list_mot_hist.append(sum_variances)

    return np.array(list_mot_hist)

def get_next_relevant_cuboids(vstream,gs=False):
    _, _, _, cuboids_batch, _ = vstream.get_next_cuboids_from_stream(gs)

    if (len(vstream.seek_dict[vstream.seek]) <= 1):
        cuboids_batch = cuboids_batch[-len(cuboids_batch) / 3:]

    return cuboids_batch

def get_next_relevant_cuboids_test(vstream,gs=False):
    list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly, _, _ = vstream.get_next_cuboids_from_stream(gs)

    return list_cuboids,list_cuboids_pixmap,list_cuboids_anomaly

def get_batch_with_motion(batch,thresh_variance,ts_pos=0,gs=0):

    variances = get_variances(batch,ts_pos,gs)
    return batch[np.where(variances>=thresh_variance)]

def process_a_batch(model,vstream,thresh_variance,mini_batch_size,train=True,gs=False,ts_pos=0):

    cuboids_batch = get_next_relevant_cuboids(vstream,gs)
    cuboids_batch = norm_batch(cuboids_batch)
    cuboids_batch = get_batch_with_motion(cuboids_batch,thresh_variance,ts_pos,gs)
    train_on_batch(model,cuboids_batch,mini_batch_size,train)

def process_k_means_a_batch(model,vstream,thresh_variance,mini_batch_size,ts_pos=0):

    cuboids_batch = get_next_relevant_cuboids(vstream)
    cuboids_batch = norm_batch(cuboids_batch)
    cuboids_batch = get_batch_with_motion(cuboids_batch,thresh_variance,ts_pos)
    kmeans_on_batch(model,cuboids_batch,mini_batch_size)

def do_recon(model,vstream,thresh_variance,num_recons,ts_pos=0):
    cuboids_batch = get_next_relevant_cuboids(vstream)
    cuboids_batch = norm_batch(cuboids_batch)
    cuboids_batch = get_batch_with_motion(cuboids_batch,thresh_variance,ts_pos)

    for i in range(0,num_recons):
        input_cuboid = cuboids_batch[np.random.randint(0,len(cuboids_batch))]
        model.do_gif_recon(input_cuboid,'recon'+str(i))

    return True

def return_relevant_cubs(vstream,thresh_variance,gs=False,ts_pos=0):

    cuboids_batch = get_next_relevant_cuboids(vstream,gs)

    if(gs==0):
        cuboids_batch = norm_batch(cuboids_batch)

    cuboids_batch = get_batch_with_motion(cuboids_batch,thresh_variance,ts_pos,gs)

    return cuboids_batch

def make_test_cuboid_structures(list_cuboids,list_cuboids_pixmap,list_cuboids_anomaly):

    rows = list_cuboids[0].shape[0]
    cols = list_cuboids[0].shape[1]

    cubstructs=[]

    for j in range(1, rows - 1):
        for k in range(1, cols - 1):

            surroundings = []

            surr_idx = 0

            current_cuboid = list_cuboids[1][j, k]


            surroundings.append(list_cuboids[surr_idx][j - 1, k - 1])
            surroundings.append(list_cuboids[surr_idx][j - 1, k])
            surroundings.append(list_cuboids[surr_idx][j - 1, k + 1])

            surroundings.append(list_cuboids[surr_idx][j, k - 1])
            surroundings.append(list_cuboids[surr_idx][j, k])
            surroundings.append(list_cuboids[surr_idx][j, k + 1])

            surroundings.append(list_cuboids[surr_idx][j + 1, k - 1])
            surroundings.append(list_cuboids[surr_idx][j + 1, k])
            surroundings.append(list_cuboids[surr_idx][j + 1, k + 1])


            surr_idx = 1
            surroundings.append(list_cuboids[surr_idx][j - 1, k - 1])
            surroundings.append(list_cuboids[surr_idx][j - 1, k])
            surroundings.append(list_cuboids[surr_idx][j - 1, k + 1])

            surroundings.append(list_cuboids[surr_idx][j, k - 1])
            surroundings.append(list_cuboids[surr_idx][j, k + 1])

            surroundings.append(list_cuboids[surr_idx][j + 1, k - 1])
            surroundings.append(list_cuboids[surr_idx][j + 1, k])
            surroundings.append(list_cuboids[surr_idx][j + 1, k + 1])


            surr_idx = 2
            surroundings.append(list_cuboids[surr_idx][j - 1, k - 1])
            surroundings.append(list_cuboids[surr_idx][j - 1, k])
            surroundings.append(list_cuboids[surr_idx][j - 1, k + 1])

            surroundings.append(list_cuboids[surr_idx][j, k - 1])
            surroundings.append(list_cuboids[surr_idx][j, k])
            surroundings.append(list_cuboids[surr_idx][j, k + 1])

            surroundings.append(list_cuboids[surr_idx][j + 1, k - 1])
            surroundings.append(list_cuboids[surr_idx][j + 1, k])
            surroundings.append(list_cuboids[surr_idx][j + 1, k + 1])


            cub =  Cuboid(current_cuboid,list_cuboids_pixmap[1][j][k],surroundings,list_cuboids_pixmap[1][j][k])

            cubstructs.append(cub)

    return cubstructs

def return_cuboid_test(vstream,thresh_variance,gs=False,ts_pos=0):

    #This function gets cuboids in the array format, processes and extracts neighors and checks if all the neighbors
    #have enough variance to be computed by the model. This is temporary.
    # TODO. Fix this to a more permanent one after analysis

    list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly = get_next_relevant_cuboids_test(vstream,gs)

    if(gs==0):
        for idx,i in enumerate(list_cuboids):
            list_cuboids[idx] = norm_batch(i)

    cubstructs = make_test_cuboid_structures(list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly)

    return cubstructs
