import numpy as np
from sklearn.utils import shuffle


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

def get_variances(cuboids_batch,ts_pos=0):

    list_mot_hist=[]

    for j in cuboids_batch:
        sum_variances = np.mean(np.var(j,axis=ts_pos))
        list_mot_hist.append(sum_variances)

    return np.array(list_mot_hist)

def get_next_relevant_cuboids(vstream,gs=False):
    _, _, _, cuboids_batch, _ = vstream.get_next_cuboids_from_stream(gs)

    if (len(vstream.seek_dict[vstream.seek]) <= 1):
        cuboids_batch = cuboids_batch[-len(cuboids_batch) / 3:]

    return cuboids_batch

def get_batch_with_motion(batch,thresh_variance,ts_pos=0):

    variances = get_variances(batch,ts_pos)
    return batch[np.where(variances>=thresh_variance)]

def process_a_batch(model,vstream,thresh_variance,mini_batch_size,train=True,gs=False,ts_pos=0):

    cuboids_batch = get_next_relevant_cuboids(vstream,gs)
    cuboids_batch = norm_batch(cuboids_batch)
    cuboids_batch = get_batch_with_motion(cuboids_batch,thresh_variance,ts_pos)
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
    cuboids_batch = norm_batch(cuboids_batch)
    cuboids_batch = get_batch_with_motion(cuboids_batch,thresh_variance,ts_pos)

    return cuboids_batch

