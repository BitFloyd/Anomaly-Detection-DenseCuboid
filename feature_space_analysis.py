import numpy as np
from scipy import stats, integrate
from sklearn.preprocessing import StandardScaler,Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os
from matplotlib.backends.backend_pdf import PdfPages

sns.set(color_codes=True)

model_store = '/usr/local/data/sejacob/ANOMALY/densecub/models/nocl_tstrd_4_nic_0_chapters_0_clusters_15_hunits_64_color__rev_True_large_True_ntrain_100_lamda_0.0'

features_h5 = h5py.File(os.path.join(model_store, 'features.h5'), 'r')

list_feats = []

for id in range(0, len(features_h5)):
    f_arr = np.array(features_h5.get('chapter_' + str(id)))
    list_feats.extend(f_arr.tolist())
    del f_arr

features_h5.close()

list_feats = np.array(list_feats)

plots_done = 0

pdf_name = os.path.join(model_store,'features_kde.pdf')

rows_plots = (list_feats.shape[1]/8)

with PdfPages(pdf_name) as pdf:

    for j in range(0,3):

        fig, ax = plt.subplots(ncols=8, nrows=rows_plots, figsize=(40,40),sharex='col', sharey='row')

        if(j==0):
            list_feats_to_plot = list_feats
            plt.suptitle('Features as is without any scaling')
        elif(j==1):
            sc = Normalizer()
            list_feats_to_plot = sc.fit_transform(list_feats)
            plt.suptitle('Features - Normalized')
        else:
            sc = StandardScaler()
            list_feats_to_plot = sc.fit_transform(list_feats)
            plt.suptitle('Features - Scaled')


        for i in range(0, list_feats_to_plot.shape[1]):

            ax[int(i / rows_plots)][i % 8].set_title('feature: ' + str(i + 1))
            sns.distplot(list_feats_to_plot[:, i], kde=True, rug=False, hist=True, ax=ax[int(i / rows_plots)][i % 8])


        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

