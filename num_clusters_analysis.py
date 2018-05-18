import matplotlib as mpl
mpl.use('Agg')
from functionals_pkg import argparse_fns as af
from sys import argv
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
sns.set(color_codes=True)
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import MiniBatchKMeans
import matplotlib.cm as cm
from tqdm import  tqdm


def kmeans_and_update_pdf(n_clusters=10,pdf=None):

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

    ax.set_title("The silhouette plot for the various clusters",fontsize=20)
    ax.set_xlabel("The silhouette coefficient values",fontsize=20)
    ax.set_ylabel("Cluster label",fontsize=20)

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.axvline(x=0.0, color="black", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks(list(np.round(np.linspace(-1,1,20),decimals=1)))

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                  fontsize=30)

    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    return silhouette_avg


metric = af.getopts(argv)

rdict = af.parse_run_variables(metric)

n_gpus = rdict['n_gpus']
guill = rdict['guill']
data_store_suffix = rdict['data_store_suffix']
batch_size=rdict['batch_size']
n_chapters = rdict['n_chapters']
gs = rdict['gs']
nic = rdict['nic']
tstrides = rdict['tstrides']
lassign = rdict['lassign']
h_units = rdict['h_units']
loss = rdict['loss']
ntrain = rdict['ntrain']
nclusters = rdict['nclusters']
nocl = rdict['nocl']
lamda = rdict['lamda']
train_folder = rdict['train_folder']
test_data_store = rdict['test_data_store']
nc=rdict['nc']
large = rdict['large']
tlm = rdict['tlm']
reverse = rdict['reverse']
model_store = rdict['model_store']
size = 24

print "################################"
print "LOAD_FEATURES"
print "################################"

features_h5 = h5py.File(os.path.join(model_store, 'features.h5'), 'r')

list_feats = []

for id in range(0, n_chapters):
    f_arr = np.array(features_h5.get('chapter_' + str(id)))
    list_feats.extend(f_arr.tolist())
    del f_arr


list_feats = np.array(list_feats)


print "################################"
print "FEATURES_SHAPE:", list_feats.shape
print "################################"
range_n_clusters = range(5,205,5)
print "################################"
print "RANGE:", range_n_clusters
print "################################"

pdf_name = os.path.join(model_store, 'silhouette_analysis.pdf')

select_indexes = np.random.randint(0,len(list_feats),int(0.01*len(list_feats)))

if(os.path.exists(pdf_name)):
    os.remove(pdf_name)

silhouette_avg_score_list = []
with PdfPages(pdf_name) as pdf:
    for n_clusters in tqdm(range_n_clusters):

        avg_score = kmeans_and_update_pdf(n_clusters,pdf)

        silhouette_avg_score_list.append(avg_score)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20,20))

    sns.regplot(x=np.array(range_n_clusters), y=np.array(silhouette_avg_score_list), ax=ax,fit_reg=False,color='r')
    ax.set_xlabel('N_clusters')
    ax.set_ylabel('Average silhouette score')

    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()