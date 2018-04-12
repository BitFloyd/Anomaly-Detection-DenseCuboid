import numpy as np
import os
from skimage.io import imread
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import color,img_as_float
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.metrics import accuracy_score,confusion_matrix
from collections import deque
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import copy
import sys
import pickle
import math
import imageio
import h5py


list_anom_percentage = []

class TestDictionary:

    model = None
    dictionary_words = {}
    means = None

    data_store = None    #Where the test data is stored

    vid = None           #Id of the video being processed ATM -> Always points to the next id to be loaded
    cubarray = None      #Array of all cuboids sampled in a video, arranged in a spatio-temporal fashion
    amongtarray = None   #Contains anomaly groundtruths of corresponding cuboids in cubarray (shape = cubarray.shape())
    pixmaparray = None   #Each element contains a tuple of the pixel centres of corresponding cuboid in cubarray
                         # (shape = cubarray.shape())

    cubarray_process_index = None #The index of the set in cubarray that is being processed.

    list_of_cub_words_full_dataset = []
    list_of_cub_loss_full_dataset = []
    list_of_cub_anom_gt_full_dataset = []
    list_full_dset_cuboid_frequencies = []
    list_anom_gifs=[]
    list_full_dset_dist = []

    def __init__(self,model,data_store,data_test_h5=None,notest=False,model_store=None,test_loss_metric='dssim',use_dist_in_word=False,round_to=1):

        self.notest = notest

        self.timesteps = 8
        self.test_loss_metric = test_loss_metric
        self.data_store = data_store

        self.data_vc = data_test_h5[0]
        self.data_va = data_test_h5[1]
        self.data_vp = data_test_h5[2]

        self.use_dist_in_word=use_dist_in_word
        self.round_to=round_to

        if(not notest):
            self.model_enc = model.encoder
            self.model_ae = model.ae
            self.model_store = model.model_store

            self.means = model.means  #means of each cluster learned by the model

             #location of test data
            self.vid = 1

        if(notest):

            self.n_channels = 3

            self.model_store = model_store

            self.dictionary_words = dict(zip(self.load_h5data('dictionary_keys'),self.load_h5data('dictionary_values')))

            self.list_of_cub_words_full_dataset = self.load_h5data('list_cub_words_full_dataset')

            self.list_of_cub_anom_gt_full_dataset = self.load_h5data('list_cub_anomgt_full_dataset')

            self.list_of_cub_loss_full_dataset = self.load_h5data('list_cub_loss_full_dataset')

            self.list_full_dset_dist = self.load_h5data('list_dist_measure_full_dataset')

            if(os.path.exists(os.path.join(self.model_store, 'list_cub_frequencies_full_dataset.h5'))):
                self.list_full_dset_cuboid_frequencies = self.load_h5data('list_cub_frequencies_full_dataset')

    def load_data(self):

        # print os.path.join(self.data_store,'video_cuboids_array_'+str(self.vid)+'.npy')
        if('video_cuboids_array_'+str(self.vid) in self.data_vc.keys()):
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$"
            print "LOADING VIDEO ARRAY ", self.vid
            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            self.cubarray = np.array(self.data_vc.get('video_cuboids_array_'+str(self.vid)))
            self.anomgtarray = np.array(self.data_va.get('video_anomgt_array_' + str(self.vid)))
            self.pixmaparray = np.array(self.data_vp.get('video_pixmap_array_' + str(self.vid)))

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
            relevant_row_pixmap = self.pixmaparray[self.cubarray_process_index]

            self.cubarray_process_index+=1

            return (three_rows_cubarray,relevant_row_anom_gt,relevant_row_pixmap)
        else:
            return False

    def create_surroundings(self,three_rows_cubarray,relevant_row_anom_gt,gif=False):

        rows = three_rows_cubarray[0].shape[0]
        cols = three_rows_cubarray[0].shape[1]

        list_surrounding_cuboids_from_three_rows = []
        sublist_full_dataset_anom_gt = []
        sublist_full_dataset_dssim_loss = []
        sublist_full_dataset_distances = []

        for j in range(1, rows - 1):
            for k in range(1, cols - 1):
                surroundings = []

                surr_idx = 0

                current_cuboid = three_rows_cubarray[1][j, k]

                sublist_full_dataset_anom_gt.append(relevant_row_anom_gt[j,k])

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

                d = np.min(cdist(self.model_enc.predict(np.array(surroundings)),self.means),axis=1)

                sublist_full_dataset_distances.append(d)

                list_surrounding_cuboids_from_three_rows.append(np.array(surroundings))


        return np.array(list_surrounding_cuboids_from_three_rows), sublist_full_dataset_anom_gt, \
               sublist_full_dataset_dssim_loss, sublist_full_dataset_distances

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

    def update_dict_from_data(self):

        while (self.load_data()):
            self.update_dict_from_video()

        self.save_h5data('dictionary_keys',self.dictionary_words.keys())
        self.save_h5data('dictionary_values',self.dictionary_words.values())

        self.save_h5data('list_cub_words_full_dataset',self.list_of_cub_words_full_dataset)

        self.save_h5data('list_cub_anomgt_full_dataset',self.list_of_cub_anom_gt_full_dataset)

        self.save_h5data('list_cub_loss_full_dataset',self.list_of_cub_loss_full_dataset)

        self.save_h5data('list_dist_measure_full_dataset',self.list_full_dset_dist)


        return True

    def update_dict_from_video(self):

        next_set_from_cubarray = self.fetch_next_set_from_cubarray()

        while(next_set_from_cubarray):

            surroundings_from_three_rows, sublist_full_dataset_anom_gt, sublist_full_dataset_dssim_loss,\
            sublist_full_dataset_distances = self.create_surroundings(three_rows_cubarray=next_set_from_cubarray[0],
                                                                             relevant_row_anom_gt=next_set_from_cubarray[1])

            predictions = self.predict_on_surroundings(surroundings_from_three_rows)
            words_from_preds = self.create_words_from_predictions(predictions,sublist_full_dataset_distances)
            self.update_dict_with_words(words_from_preds)

            self.list_of_cub_anom_gt_full_dataset.extend(sublist_full_dataset_anom_gt)
            self.list_of_cub_loss_full_dataset.extend(sublist_full_dataset_dssim_loss)
            self.list_full_dset_dist.extend(sublist_full_dataset_distances)
            self.list_of_cub_words_full_dataset.extend(words_from_preds)

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
        plt.savefig(os.path.join(self.model_store, graph_name_frq), bbox_inches='tight')
        plt.close()

        hfm, = plt.plot(self.list_of_cub_loss_full_dataset, label='loss values')
        plt.legend(handles=[hfm])
        plt.title('DSSIM Reconstruction Loss')
        plt.xlabel('Cuboid Index')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.model_store, graph_name_loss), bbox_inches='tight')
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
            self.list_full_dset_cuboid_frequencies.append(self.dictionary_words[i])


        self.save_h5data('list_cub_frequencies_full_dataset', self.list_full_dset_cuboid_frequencies)

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print "LEN LIST_CUB_FREQUENCIES_FULL", len(self.list_full_dset_cuboid_frequencies)
        print "MAX LIST_CUB_FREQUENCIES_FULL", max(self.list_full_dset_cuboid_frequencies)
        print "MIN LIST_CUB_FREQUENCIES_FULL", min(self.list_full_dset_cuboid_frequencies)
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

        return True

    def make_p_r_f_a_curve(self,prfa_graph_name,tp_fp_graph_name,deets_filename):

        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        accuracy_score_list = []

        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []

        lspace = np.logspace(math.log10(min(self.dictionary_words.values())),math.log10(max(self.dictionary_words.values())),1000)

        total_num_tp = np.sum(self.list_of_cub_anom_gt_full_dataset)

        print "#################################################################################"
        print "TOTAL NUMBER OF TRUE POSITIVES:" , total_num_tp
        print "TOTAL NUMBER OF SAMPLES TO BE TESTED:",len(self.list_of_cub_anom_gt_full_dataset)
        print "RATIO:",total_num_tp/(len(self.list_of_cub_anom_gt_full_dataset)+0.0)
        print "ACC WHEN ALL 0: ", (len(self.list_of_cub_anom_gt_full_dataset) - total_num_tp)/(len(self.list_of_cub_anom_gt_full_dataset)+0.0) * 100
        print "#################################################################################"


        for i in tqdm(lspace):

            y_true = np.array(self.list_of_cub_anom_gt_full_dataset)
            y_pred = (np.array(self.list_full_dset_cuboid_frequencies)<=i)

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
        print "F1 score:", lspace[f1_score_list.index(max(f1_score_list))]
        print "##########################################################################"

        #Plot PRFA curve
        hfm, = plt.plot(precision_score_list, label='precision_score')
        hfm2, = plt.plot(recall_score_list, label='recall_score')
        hfm3, = plt.plot(f1_score_list, label='f1_score')
        hfm4, = plt.plot(accuracy_score_list,label='accuracy_score')

        s_acc = 'max_acc:'+str(format(max(accuracy_score_list),'.2f'))+' at '+ str(int(lspace[accuracy_score_list.index(max(accuracy_score_list))]))
        s_f1 = 'max_f1:'+str(format(max(f1_score_list),'.2f'))+' at '+ str(int(lspace[f1_score_list.index(max(f1_score_list))]))
        s_pr = 'max_pr:'+str(format(max(precision_score_list),'.2f'))+' at '+ str(int(lspace[precision_score_list.index(max(precision_score_list))]))
        s_re = 'max_re:' + str(format(max(recall_score_list),'.2f')) + ' at ' + str(int(lspace[recall_score_list.index(max(recall_score_list))]))


        plt.legend(handles=[hfm,hfm2,hfm3,hfm4])
        plt.title('Precision, Recall, F1_Score')
        plt.ylabel('Scores')
        plt.xlabel('Word frequency threshold')
        plt.savefig(os.path.join(self.model_store, prfa_graph_name), bbox_inches='tight')
        plt.close()

        f = open(os.path.join(self.model_store, deets_filename), 'a+')
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
        plt.xlabel('Word frequency threshold')
        plt.savefig(os.path.join(self.model_store, tp_fp_graph_name), bbox_inches='tight')
        plt.close()

    def make_p_r_f_a_curve_dss(self,prfa_graph_name_loss,tp_fp_graph_name_loss,deets_filename_loss):

        precision_score_list = []
        recall_score_list = []
        f1_score_list = []
        accuracy_score_list = []

        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []

        lspace = np.linspace(min(self.list_of_cub_loss_full_dataset),max(self.list_of_cub_loss_full_dataset),1000)

        total_num_tp = np.sum(self.list_of_cub_anom_gt_full_dataset)

        print "#################################################################################"
        print "TOTAL NUMBER OF TRUE POSITIVES:" , total_num_tp
        print "TOTAL NUMBER OF SAMPLES TO BE TESTED:",len(self.list_of_cub_anom_gt_full_dataset)
        print "RATIO:",total_num_tp/(len(self.list_of_cub_anom_gt_full_dataset)+0.0)
        print "ACC WHEN ALL 0: ", (len(self.list_of_cub_anom_gt_full_dataset) - total_num_tp)/(len(self.list_of_cub_anom_gt_full_dataset)+0.0) * 100
        print "#################################################################################"


        for i in tqdm(lspace):

            y_true = np.array(self.list_of_cub_anom_gt_full_dataset)
            y_pred = (np.array(self.list_of_cub_loss_full_dataset)>=i)

            cm = confusion_matrix(y_true, y_pred)

            precision_score_list.append(precision_score(y_true, y_pred) * 100)
            recall_score_list.append(recall_score(y_true, y_pred) * 100)
            f1_score_list.append(f1_score(y_true, y_pred) * 100)
            accuracy_score_list.append(accuracy_score(y_true, y_pred) * 100)

            TN = cm[0][0]

            FP = cm[0][1]

            FN = cm[1][0]

            TP = cm[1][1]

            tp_list.append(TP)
            tn_list.append(TN)

            fn_list.append(FN)
            fp_list.append(FP)

        print "##########################################################################"
        print "MAX ACCURACY:", max(accuracy_score_list)
        print "THRESHOLD:", lspace[accuracy_score_list.index(max(accuracy_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX PRECISION:", max(precision_score_list)
        print "THRESHOLD:", lspace[precision_score_list.index(max(precision_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX RECALL:", max(recall_score_list)
        print "THRESHOLD:", lspace[recall_score_list.index(max(recall_score_list))]
        print "##########################################################################"

        print "##########################################################################"
        print "MAX F1:", max(f1_score_list)
        print "F1 score:", lspace[f1_score_list.index(max(f1_score_list))]
        print "##########################################################################"

        # Plot PRFA curve
        hfm, = plt.plot(precision_score_list, label='precision_score')
        hfm2, = plt.plot(recall_score_list, label='recall_score')
        hfm3, = plt.plot(f1_score_list, label='f1_score')
        hfm4, = plt.plot(accuracy_score_list, label='accuracy_score')

        s_acc = 'max_acc:' + str(format(max(accuracy_score_list), '.2f')) + ' at ' + str(
            (lspace[accuracy_score_list.index(max(accuracy_score_list))]))
        s_f1 = 'max_f1:' + str(format(max(f1_score_list), '.2f')) + ' at ' + str(
            (lspace[f1_score_list.index(max(f1_score_list))]))
        s_pr = 'max_pr:' + str(format(max(precision_score_list), '.2f')) + ' at ' + str(
            (lspace[precision_score_list.index(max(precision_score_list))]))
        s_re = 'max_re:' + str(format(max(recall_score_list), '.2f')) + ' at ' + str(
            (lspace[recall_score_list.index(max(recall_score_list))]))

        plt.legend(handles=[hfm, hfm2, hfm3, hfm4])
        plt.title('Precision, Recall, F1_Score')
        plt.ylabel('Scores')
        plt.xlabel('Word frequency threshold')
        plt.savefig(os.path.join(self.model_store, prfa_graph_name_loss), bbox_inches='tight')
        plt.close()

        f = open(os.path.join(self.model_store, deets_filename_loss), 'a+')
        f.write(s_acc + '\n')
        f.write(s_f1 + '\n')
        f.write(s_pr + '\n')
        f.write(s_re + '\n')
        f.close()

        # Plot TPFPcurve
        hfm, = plt.plot(tp_list, label='N_true_positives')
        hfm2, = plt.plot(fp_list, label='N_false_positives')
        hfm3, = plt.plot(tn_list, label='N_true_negatives')
        hfm4, = plt.plot(fn_list, label='N_false_negatives')

        plt.legend(handles=[hfm, hfm2, hfm3, hfm4])
        plt.title('TP,FP,TN,FN')
        plt.ylabel('Values')
        plt.xlabel('Word frequency threshold')
        plt.savefig(os.path.join(self.model_store, tp_fp_graph_name_loss), bbox_inches='tight')
        plt.close()

    def make_comparitive_plot(self,graph_name,array_to_consider,metric_name=None):

        y_true = np.array(self.list_of_cub_anom_gt_full_dataset)

        args_arr_sort = np.argsort(array_to_consider)

        y_true_arr = y_true[args_arr_sort]

        array_to_consider = array_to_consider[args_arr_sort]

        colors = ['green', 'red']

        f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(40, 30))

        im1 = ax1.scatter(range(0,len(array_to_consider)),array_to_consider,c=y_true_arr, cmap=ListedColormap(colors),alpha=0.5)
        ax1.set_title('ANOMS:Red, N-ANOMS:Green')
        ax1.set_ylabel(metric_name)
        ax1.set_xlabel('Cuboid index')
        ax1.grid(True)
        cb1 = f.colorbar(im1,ax=ax1)
        loc = np.arange(0, max(y_true), max(y_true) / float(len(colors)))
        cb1.set_ticks(loc)
        cb1.set_ticklabels(['normal','anomaly'])

        arr_anoms = array_to_consider[y_true_arr==1]

        im2 = ax2.scatter(range(0,len(arr_anoms)),arr_anoms,c='red',alpha=0.5)
        ax2.set_title('ANOMS:Red')
        ax2.set_ylabel(metric_name)
        ax2.set_xlabel('Cuboid index')
        ax2.grid(True)

        plt.savefig(os.path.join(self.model_store, graph_name), bbox_inches='tight')
        plt.close()

        return True

    def plot_frequencies_of_samples(self,anom_frequency_graph_name):

        frequency_array = np.array(self.list_full_dset_cuboid_frequencies)
        self.make_comparitive_plot(anom_frequency_graph_name,frequency_array,'Frequency')

        return True

    def plot_loss_of_samples(self,recon_loss_graph_name):

        loss_array = np.array(self.list_of_cub_loss_full_dataset)
        self.make_comparitive_plot(recon_loss_graph_name,loss_array,self.test_loss_metric+'-loss')

        return True

    def plot_distance_measure_of_samples(self,distance_measure_samples_graph_name,dmeasure='mean'):

        if(dmeasure=='mean'):
            dmeasure_array = np.mean(np.array(self.list_full_dset_dist),axis=1)

        elif(dmeasure=='std'):
            dmeasure_array = np.std(np.array(self.list_full_dset_dist),axis=1)

        elif(dmeasure=='meanxloss'):
            dmeasure_array = np.mean(np.array(self.list_full_dset_dist),axis=1) * np.array(self.list_of_cub_loss_full_dataset)

        elif(dmeasure=='stdxloss'):
            dmeasure_array = np.std(np.array(self.list_full_dset_dist), axis=1) * np.array(self.list_of_cub_loss_full_dataset)

        else:
            print "ERROR: DMEASURE MUST BE = mean or std"
            return False

        self.make_comparitive_plot(distance_measure_samples_graph_name,dmeasure_array,'distance-measure')

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
        self.create_pdf_distance_surroundings(list_distances=list_normal_cuboid_distances,pdf_name=os.path.join(self.model_store,'cubds_norm_dist_bar.pdf'))

        # distance_anom_cuboids_pdf
        list_anom_cuboid_distances = np.array(self.list_full_dset_dist)[np.array(self.list_of_cub_anom_gt_full_dataset)==1]
        list_anom_cuboid_distances = list_anom_cuboid_distances[np.random.randint(0,len(list_anom_cuboid_distances),num_plots_per_pdf)]
        self.create_pdf_distance_surroundings(list_distances=list_anom_cuboid_distances,pdf_name=os.path.join(self.model_store,'cubds_anom_dist_bar.pdf'))

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
            surroundings_from_three_rows, sublist_full_dataset_anom_gt, _ = \
                self.create_surroundings(three_rows_cubarray=next_set_from_cubarray[0],relevant_row_anom_gt=next_set_from_cubarray[1]
                                         ,gif=True)

            if(not any(sublist_full_dataset_anom_gt)):
                next_set_from_cubarray = self.fetch_next_set_from_cubarray()
                continue
            else:
                surroundings_from_three_rows = surroundings_from_three_rows[np.array(sublist_full_dataset_anom_gt)==True]
                predictions = self.predict_on_surroundings(surroundings_from_three_rows)
                words_from_preds = self.create_words_from_predictions(predictions)

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

    def __init__(self, video_path,video_train_test, size_y,size_x,timesteps,num=-1,ts_first_or_last='first',strides=1,tstrides=1,anompth=0.0,bkgsub=False):

        # Initialize-video-params
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

            list_cuboids, list_cuboids_pixmap, list_cuboids_anomaly, list_all_cuboids, list_all_cuboids_gt = \
            make_cuboids_for_stream(self,self.seek_dict[self.seek], self.seek_dict_gt[self.seek], self.size_x, self.size_y,
                                    test_or_train=self.video_train_test, ts_first_last = self.ts_first_or_last,strides=self.strides,gs=gs,anompth=self.anompth,
                                    bkgsub=self.bkgsub)

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

def make_cuboids_for_stream(stream,list_images,list_images_gt,size_x,size_y,test_or_train='Train', ts_first_last = 'first',strides=1,gs=True,anompth=0.0,bkgsub=False):


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

                    anompercentage = anomaly_gt_sum/((end_cols-start_cols)*(end_rows-start_rows)*n_channels*n_frames*255.0)

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

def mean_squared_error(x,y):
    return np.sqrt(np.mean((x-y)**2))
