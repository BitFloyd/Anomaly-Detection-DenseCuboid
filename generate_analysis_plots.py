import matplotlib.pyplot as plt
import numpy as np

#Clusters vs precision
n_clusters = [10,15,25,50]
n_c_nocl1_pre =  [3.71,5.50,2.30,3.12]
n_c_nocl2_pre =  [3.02,3.37,5.20,0.00]
n_c_l_0_01_pre = [5.10,6.46,0.00,0.00]
n_c_l_0_1_pre =  [9.47,6.42,5.11,2.01]

#Clusters vs unique words
n_c_nocl1_uq_words =  [7000,7100,35000,240000]
n_c_nocl2_uq_words =  [8000,7200,49000,0000]
n_c_l_0_01_uq_words = [14000,6300,0000,0000]
n_c_l_0_1_uq_words =  [18000,29000,160000,600000]

#Epochs vs precision
#10_clusters
n_epochs_10_c =  [200,250,300,400]
precision_10_c = [3.71,3.02,5.10,9.47]

#15_clusters
n_epochs_15_c =  [300,200,340,300]
precision_15_c = [5.50,3.37,6.42,6.46]

#25_clusters
n_epochs_25_c =  [300,200,375,0]
precision_25_c = [2.3,5.2,5.11,0]

#50_clusters
n_epochs_50_c =  [300,450]
precision_50_c = [3.12,2.01]

f, (ax1,ax2,ax3) = plt.subplots(3,figsize=(8.27,11.69))

hfm_nocl1_pre  = ax1.scatter(n_clusters,n_c_nocl1_pre,label='no-cl1')
ax1.plot(n_clusters,n_c_nocl1_pre,':')
hfm_nocl2_pre  = ax1.scatter(n_clusters,n_c_nocl2_pre,label='no-cl2')
ax1.plot(n_clusters,n_c_nocl2_pre,':')
hfm_l_0_01_pre = ax1.scatter(n_clusters,n_c_l_0_01_pre,label='l_0_01')
ax1.plot(n_clusters,n_c_l_0_01_pre,':')
hfm_l_0_1_pre  = ax1.scatter(n_clusters,n_c_l_0_1_pre, label='l_0_1')
ax1.plot(n_clusters,n_c_l_0_1_pre,':')
ax1.legend(handles=[hfm_nocl1_pre,hfm_nocl2_pre,hfm_l_0_01_pre,hfm_l_0_1_pre])
ax1.grid(True)
ax1.set_title('CLUSTERS vs PRECISION')
ax1.set_ylabel('precision')
ax1.set_xlabel('clusters')

hfm_nocl1_uq = ax2.scatter(n_clusters,n_c_nocl1_uq_words, label='no-cl1')
ax2.plot(n_clusters,n_c_nocl1_uq_words,':')
hfm_nocl2_uq = ax2.scatter(n_clusters,n_c_nocl2_uq_words, label='no-cl2')
ax2.plot(n_clusters,n_c_nocl2_uq_words,':')
hfm_l_0_01_uq = ax2.scatter(n_clusters,n_c_l_0_01_uq_words, label='l_0_01')
ax2.plot(n_clusters,n_c_l_0_01_uq_words,':')
hfm_l_0_1_uq = ax2.scatter(n_clusters,n_c_l_0_1_uq_words, label='l_0_1')
ax2.plot(n_clusters,n_c_l_0_1_uq_words,':')
ax2.legend(handles=[hfm_nocl1_uq,hfm_nocl2_uq,hfm_l_0_01_uq,hfm_l_0_1_uq])
ax2.grid(True)
ax2.set_title('CLUSTERS vs UNIQUE WORDS')
ax2.set_ylabel('n_unique_words')
ax2.set_xlabel('clusters')


hfm_10_c_ep = ax3.scatter(n_epochs_10_c,precision_10_c, label='10_c')
ax3.plot(n_epochs_10_c,precision_10_c,':')
hfm_15_c_ep = ax3.scatter(n_epochs_15_c,precision_15_c, label='15-c')
ax3.plot(n_epochs_15_c,precision_15_c,':')
hfm_25_c_ep = ax3.scatter(n_epochs_25_c,precision_25_c, label='25-c')
ax3.plot(n_epochs_25_c,precision_25_c,':')
hfm_50_c_ep = ax3.scatter(n_epochs_50_c,precision_50_c, label='50-c')
ax3.plot(n_epochs_50_c,precision_50_c,':')
ax3.legend(handles=[hfm_10_c_ep,hfm_15_c_ep,hfm_25_c_ep,hfm_50_c_ep])
ax3.grid(True)
ax3.set_title('EPOCHS vs PRECISION')
ax3.set_ylabel('Precision')
ax3.set_xlabel('Epochs')

f.subplots_adjust(hspace=0.3)
plt.savefig("foo.eps", orientation = 'portrait', format = 'eps', bbox_inches='tight')
plt.close()