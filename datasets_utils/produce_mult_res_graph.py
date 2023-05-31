"""Used to calculate AP from masks obtained through original Cellpose model"""

import os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.legend_handler import HandlerTuple
import seaborn
# seaborn.set()


path = '/mnt/12F9CADD61CB0337/results/cell_analysis/new_cellTranspose_morphology_loss_pretrained_adaptation_results/Adaptation/TissueNet/3-shot'
    
AP=[]
F1=[]
file_dict_AP = {}
files = os.listdir(path)
for i in files:
    with open (os.path.join(path,i), 'rb') as f:
        AP_pkl = pk.load(f,encoding='utf-8')
    mean_AP, mean_F1, _, _ = AP_pkl
    file_dict_AP[i] = mean_AP[51]
    
sorted_AP = dict(sorted(file_dict_AP.items(), key=lambda x:x[1]))

for i in list(sorted_AP.keys()):
    with open (os.path.join(path,i), 'rb') as f:
        AP_pkl = pk.load(f,encoding='utf-8')
    mean_AP, mean_F1, _, _ = AP_pkl
    AP.append(mean_AP[50:])
    F1.append(mean_F1[50:])

tau = np.arange(0.5, 1.01, 0.01)
plt.figure()
plt.title('Average Precision for TissueNet')
plt.xlabel(r'IoU Matching Threshold $\tau$')
plt.ylabel('Average Precision')
plt.yticks(np.arange(0, 1.01, step=0.2))

weights = np.arange(1, 14)
norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
cmap1 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap2 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)


plots = []
legends = []
plt.text(0.75,0.9 ,'CellTranspose', fontsize = 10, color=cmap1.to_rgba(14))
plt.text(0.9,0.9 ,'Cellpose', fontsize = 10, color=cmap2.to_rgba(14))
for i in range(len(AP)):
    a, = plt.plot(tau, AP[i], color=cmap1.to_rgba(i+1), marker='', fillstyle='left')
    b, = plt.plot(tau, AP[i], color=cmap2.to_rgba(i+1), marker='', fillstyle='right')
    plots.append((a,b))
    legends.append(os.path.splitext(list(sorted_AP.keys())[i])[0])
    
plt.legend((plots), (legends), numpoints=1, fontsize=9,  labelspacing=-2, 
           frameon=False, loc='center left', handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=3)
plt.savefig('AP Results')

# plt.figure()
# plt.plot(tau, f1_overall)
# plt.title('F1 Score for TissueNet')
# plt.xlabel(r'IoU Matching Threshold $\tau$')
# plt.ylabel('F1 Score')
# plt.yticks(np.arange(0, 1.01, step=0.2))
# plt.savefig(os.path.join(args.mask_path, 'F1 Score'))

# print('AP Results at IoU threshold 0.5: AP = {}\nF1 score at IoU threshold 0.5: F1 = {} \nTrue Postive: {}; False Positive: {}; '
#     'False Negative: {}'.format(ap_overall[51], f1_overall[51], tp_overall[51], fp_overall[51], fn_overall[51]))

