import os
import numpy as np
import pickle as pk 
from tqdm import tqdm
import operator
import matplotlib.pyplot as plt
import argparse
import tifffile as tiff

def find_top_x(path, x=5):
    
    if len(os.listdir(path)) < x: 
        print("x is larger than the number of resuts in the dir")
        return
    
    res = {}
    for version in tqdm(os.listdir(path)):
        files_path = os.path.join(path, version)
        
        if os.path.isdir(files_path):
            for file in os.listdir(files_path):
                
                if file.endswith('AP_Results.pkl'):
                    # print(f'{file}')
                    with open (os.path.join(files_path,file), 'rb') as f:
                        tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error = pk.load(f,encoding='utf-8')
                    
                    res[os.path.join(files_path,file)]=ap_overall[51]
                    # print(f'{version}:    {ap_overall[51]:.4f}')
    
    sortedList = sorted(res.items(), key=operator.itemgetter(1),reverse=True)[:x]
    
    for i in sortedList: 
        print(f'{i[0]}: {i[1]}')

    return sortedList

def calc_avg_std(sorted_list, save_path=None, masks_saved=False):
    
    if save_path is not None and os.path.exists(save_path) == False: 
        os.makedirs(save_path)
        
    AP=[]
    F1=[]
    TAU=[]
    
    for i in sorted_list:  
        with open (os.path.join(i[0]), 'rb') as f:
            tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error = pk.load(f,encoding='utf-8')
            f1_overall = tp_overall / (tp_overall + 0.5 * (fp_overall+ fn_overall))
        AP.append(ap_overall)
        F1.append(f1_overall)
        TAU = tau
        
    mean_AP = np.average(AP,axis=0)
    mean_F1 = np.average(F1,axis=0)
    
    std_AP = np.std(AP,axis=0)
    std_F1 = np.std(F1,axis=0)
    
    if masks_saved:
        masks = calc_avg_masks(sorted_list)
    else:
        masks = '-'
        
    plt.figure()
    plt.plot(TAU, mean_AP)
    plt.fill_between(TAU, mean_AP - std_AP, mean_AP + std_AP, color='#888888', alpha=0.4)
    plt.title(f'Average Precision (averaged over 5) - AP@0.5: {mean_AP[51]:.3f}\nmasks {masks}')
    plt.xlabel(r'IoU Matching Threshold $\tau$')
    plt.ylabel('Average Precision')
    plt.yticks(np.arange(0, 1.01, step=0.2))
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'AVG AP Results'))
        
    plt.figure()
    plt.plot(TAU, mean_F1)
    plt.fill_between(TAU, mean_F1- std_F1, mean_F1 + std_F1, color='#888888', alpha=0.4)
    plt.title(f'F1 Score (averaged over 5) - F1@0.5 {mean_F1[51]:.3f}\nmasks {masks}')
    plt.xlabel(r'IoU Matching Threshold $\tau$')
    plt.ylabel('F1 Score')
    plt.yticks(np.arange(0, 1.01, step=0.2))
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'AVG F1 Score'))

def calc_avg_masks(sorted_list):
   
    total_masks=[]
    for i in sorted_list:
        dir_path = os.path.dirname(i)
        for curr_path in os.listdir(dir_path):
            curr_dir = os.path.join(dir_path, curr_dir)
            if os.path.isdir(curr_dir) and curr_path.endswith('shot'):
                l_dir = os.path.join(curr_dir,'labels')
                for tiff_mask in l_dir:
                    total_masks.append(np.unique(tiff.imread(os.path.join(l_dir,tiff_mask))[1:]))
            
    return int(np.average(total_masks))
       
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-path')
    parser.add_argument('--save-path',default=None)
    args = parser.parse_args()

    sorted_list = find_top_x(args.res_path)
    
    calc_avg_std(sorted_list,args.save_path)
    
    
