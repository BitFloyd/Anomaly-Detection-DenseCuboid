import matplotlib as mpl
mpl.use('Agg')
import model_pkg.models as models
from functionals_pkg import argparse_fns as af
from functionals_pkg import save_objects as so
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


def save_pdf(index=0):

    if(os.path.exists(pdf_name)):
        os.remove(pdf_name)

    with PdfPages(pdf_name) as pdf:

        for indx,j in enumerate(score_dict_running.keys()):

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(40,40))

            plt.suptitle(list_titles[j])
            print "J = ", j
            print "LIST_COMPONENTS = " ,list_components[0:index+1]
            print "SCORE_DICT_RUNNING = ", score_dict_running[j]

            sns.regplot(x=np.array(np.log(list_components[0:index + 1])), y=np.array(score_dict_running[j]), ax=ax)
            ax.set_xlabel('Log Scale: N_components',fontsize=30)
            ax.set_ylabel('Score',fontsize=30)

            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

    return True

def run_and_update(comps = 10):

    global avg_time_taken

    # Get Test class
    data_h5_vc = h5py.File(os.path.join(test_data_store, 'data_test_video_cuboids.h5'))
    data_h5_va = h5py.File(os.path.join(test_data_store, 'data_test_video_anomgt.h5'))
    data_h5_vp = h5py.File(os.path.join(test_data_store, 'data_test_video_pixmap.h5'))
    data_h5_ap = h5py.File(os.path.join(test_data_store, 'data_test_video_anomperc.h5'))

    start_time = time.time()

    print "######################################"
    print "TESTING NOW FOR ", comps, " COMPONENTS"
    print "######################################"
    notest = False
    udiw = False

    if(notest):
        ae_model = None
    else:
        ae_model = models.Conv_autoencoder_nostream(model_store=model_store, size_y=24, size_x=24, n_channels=3, h_units=h_units,
                                            n_timesteps=8, loss=loss, batch_size=batch_size, n_clusters=nclusters,
                                            lr_model=1e-3, lamda=lamda,gs=gs,notrain=True,
                                            reverse=reverse, data_folder=train_folder,dat_h5=None,large=large)

        ae_model.perform_dict_learn(n_chapters=n_chapters,guill=guill,n_comp=comps)
        ae_model.set_cl_loss(0.0)


    use_basis_dict = True

    tclass = TestDictionary(ae_model,data_store=test_data_store,data_test_h5=[data_h5_vc,data_h5_va,data_h5_vp,data_h5_ap],
                        notest=notest,model_store=model_store,test_loss_metric=tlm,use_dist_in_word=udiw,
                        use_basis_dict=use_basis_dict)

    print "############################"
    print "UPDATING DICT FROM DATA"
    print "############################"
    tclass.process_data()

    score_dict = tclass.evaluate_prfa_dict_recon(comps=comps)

    for j in score_dict_running.keys():
        score_dict_running[j].append(score_dict[j])

    end_time = time.time()
    time_taken_hours = (end_time-start_time)/3600.0

    print "############################"
    print "PRINT TIME TAKEN :",time_taken_hours, "HOURS"
    print "############################"


    data_h5_vc.close()
    data_h5_va.close()
    data_h5_vp.close()
    data_h5_ap.close()

    return True

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
print "START TESTING"
print "################################"


print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
print train_folder
print test_data_store
print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

pdf_name = os.path.join(model_store,'effect_of_n_components.pdf')


list_titles = {'max_acc':'Maximum accuracy score','max_f1':'Maximum F1 score','max_pre':'Maximum Precision Score','max_rec':'Maximum Recall score'}

score_dict_running = {'max_acc':[],'max_f1':[],'max_pre':[],'max_rec':[]}


list_components = list(np.logspace(1.5,5,15).astype(int))

avg_time_taken = 0.0

for idx,i in enumerate(list_components):
    components = i
    run_and_update(comps=components)
    save_pdf(idx)






