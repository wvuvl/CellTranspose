
import pickle as pk 
import os, glob

path = '/media/ramzaveri/12F9CADD61CB0337/results/cell_analysis/morphology_loss_pretrained_adaptation_results/TNBC/random_scale_09_1.1'

for version in sorted(os.listdir(path)):
    print(f'TNBC {version} -------')
    multiple_res = os.path.join(path,version,'target_samples_train')
    
    for data in sorted(os.listdir(multiple_res)):
        files_path = os.path.join(multiple_res, data)
        
        for file in os.listdir(files_path):
            
            if file.endswith('AP_Results.pkl'):
                # print(f'{file}')
                with open (os.path.join(files_path,file), 'rb') as f:
                    tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error = pk.load(f,encoding='utf-8')

                print(f'{version}   -   {data}:    {ap_overall[51]:.4f}')
    print()
    
    # print(f'BBB006 {version} -------')
    # multiple_res = os.path.join(path,version)
    # for data in sorted(os.listdir(multiple_res)):
    #     files_path = os.path.join(multiple_res, data)
        
    #     for file in os.listdir(files_path):
            
    #         if file.endswith('AP_Results.pkl'):
    #             # print(f'{file}')
    #             with open (os.path.join(files_path,file), 'rb') as f:
    #                 tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error = pk.load(f,encoding='utf-8')

    #             print(f'{version}   -   {data}:    {ap_overall[51]:.4f}')
    # print()
    
    # print('Platform Specific -------')
    # platform_specific = os.path.join(path,version,'tissuenet_1.0_adaptation_results_platform_specific')
    # for data in sorted(os.listdir(platform_specific)):
    #     files_path = os.path.join(platform_specific, data)
        
    #     for file in os.listdir(files_path):
            
    #         if file.endswith('AP_Results.pkl'):
    #             # print(f'{file}')
    #             with open (os.path.join(files_path,file), 'rb') as f:
    #                 tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error = pk.load(f,encoding='utf-8')

    #             print(f'{version}   -   {data}:    {ap_overall[51]:.4f}')
    # print()
    # tissue_specifc = os.path.join(path,version,'tissuenet_1.0_adaptation_results_tissue_specific')
    # print('Tissue Specific -------')
    # for data in sorted(os.listdir(tissue_specifc)):
    #     files_path = os.path.join(tissue_specifc, data)
        
    #     for file in os.listdir(files_path):
            
    #         if file.endswith('AP_Results.pkl'):
    #             # print(f'{file}')
    #             with open (os.path.join(files_path,file), 'rb') as f:
    #                 tau, ap_overall, tp_overall, fp_overall, fn_overall, false_error = pk.load(f,encoding='utf-8')

    #             print(f'{version}   -   {data}:    {ap_overall[51]:.4f}')
    # print()
    


