import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from data_pkg.data_fns import TestDictionary,TrainDictionary
from sys import argv
import os
import socket
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
sns.set(color_codes=True)
import time


def update_pdf(index=0):

    if(os.path.exists(pdf_name)):
        os.remove(pdf_name)

    with PdfPages(pdf_name) as pdf:

        for indx,j in enumerate(score_dict_running.keys()):

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))

            plt.suptitle(list_titles[indx])
            print "J = ", j
            print "LIST_COMPONENTS = " ,list_components[0:index+1]
            print "SCORE_DICT_RUNNING = ", score_dict_running[j]

            sns.regplot(x=np.array(np.log(list_components[0:index+1])), y=np.array(score_dict_running[j]),ax=ax)
            ax.set_xlabel('Log: N_components')
            ax.set_ylabel('Score')

            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

metric = af.getopts(argv)
rpd = af.parse_run_variables(metric=metric)

score_dict_running = {'max_acc':[],'max_f1':[],'max_pre':[],'max_rec':[]}

list_titles = ['Maximum accuracy score', 'Maximum F1 score', 'Maximum Precision Score', 'Maximum Recall score']

list_components = list(np.logspace(1.5,5,15).astype(int))

avg_time_taken = 0.0

pdf_name = 'effect_of_n_components.pdf'


for idx,i in enumerate(list_components):

    start_time = time.time()

    print "######################################"
    print "TESTING NOW FOR ", i, " COMPONENTS"
    print "######################################"
    notest = False
    udiw = False

    accuracy_score_list = list(np.random.rand(100))
    f1_score_list = list(np.random.rand(100))
    precision_score_list = list(np.random.rand(100))
    recall_score_list = list(np.random.rand(100))

    print "############################"
    print "MAKE BASIS_DICT_RECON SAMPLES PLOT"
    print "############################"
    score_dict = {'max_acc':float(max(accuracy_score_list)),'max_f1':float(max(f1_score_list)),'max_pre':float(max(precision_score_list)),'max_rec':float(max(recall_score_list))}

    print score_dict

    for j in score_dict_running.keys():
        score_dict_running[j].append(score_dict[j])

    end_time = time.time()
    time_taken_hours = (end_time-start_time)/3600.0

    print "############################"
    print "PRINT TIME TAKEN :",time_taken_hours, "HOURS"
    print "############################"

    avg_time_taken = (avg_time_taken*idx + time_taken_hours)/(idx+1)

    print "############################"
    print "ESTIMATED_TIME_LEFT:",avg_time_taken*(len(list_components)-1-idx), "HOURS"
    print "############################"

    update_pdf(index=idx)







