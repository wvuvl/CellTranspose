import os, shutil
from tqdm import tqdm

def convert_to_cp_format(d_path, l_path, dest_path):
    if os.path.exists(dest_path)==False: os.makedirs(dest_path)

    d_list = sorted(os.listdir(d_path))
    l_list = sorted(os.listdir(l_path))

    for d,l in tqdm(zip(d_list, l_list)):
        f_name = os.path.splitext(d)
        l_name = f_name[0]+'_masks'+f_name[-1]
        shutil.copyfile(os.path.join(d_path,d), os.path.join(dest_path,d))
        shutil.copyfile(os.path.join(l_path,l), os.path.join(dest_path,l_name))
    
if __name__ == '__main__':
    print()
    # for split in ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]:
    #     d_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_split_cleaned/{split}/test/data'
    #     l_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_split_cleaned/{split}/test/labels'
    #     dest_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_Split_Cellpose_format/{split}/test'
    #     convert_to_cp_format(d_path, l_path, dest_path)

    #     d_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_split_cleaned/{split}/train/data'
    #     l_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_split_cleaned/{split}/train/labels'
    #     dest_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_Split_Cellpose_format/{split}/train'
    #     convert_to_cp_format(d_path, l_path, dest_path)

    # for tissue in ["breast", "gi", "immune", "pancreas"]:
    #     for platform in ["imc", "codex", "cycif", "mibi", "vectra", "mxif"]:
            
    #         if os.path.exists(f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet/test/specialist_updated_mask/tissue_specific/{tissue}/{platform}'):
    #             d_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet/test/specialist_updated_mask/tissue_specific/{tissue}/{platform}/data'
    #             l_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet/test/specialist_updated_mask/tissue_specific/{tissue}/{platform}/labels'
    #             dest_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet_cp_format/tissue_specific/{tissue}/{platform}/test'
    #             convert_to_cp_format(d_path, l_path, dest_path)

    #             d_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet/train/specialist_updated_mask/tissue_specific/{tissue}/{platform}/data'
    #             l_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet/train/specialist_updated_mask/tissue_specific/{tissue}/{platform}/labels'
    #             dest_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet_cp_format/tissue_specific/{tissue}/{platform}/train'
    #             convert_to_cp_format(d_path, l_path, dest_path)

    # for platform in ["imc", "codex", "cycif", "mibi", "vectra", "mxif"]:
    #     for tissue in ["breast", "gi", "immune", "lung", "pancreas", "skin" ]:
        
            
    #         if os.path.exists(f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet/test/specialist_updated_mask/platform_specific/{platform}/{tissue}'):
    #             d_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet/test/specialist_updated_mask/platform_specific/{platform}/{tissue}/data'
    #             l_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet/test/specialist_updated_mask/platform_specific/{platform}/{tissue}/labels'
    #             dest_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet_cp_format/platform_specific/{platform}/{tissue}/test'
    #             convert_to_cp_format(d_path, l_path, dest_path)

    #             d_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet/train/specialist_updated_mask/platform_specific/{platform}/{tissue}/data'
    #             l_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet/train/specialist_updated_mask/platform_specific/{platform}/{tissue}/labels'
    #             dest_path = f'/mnt/26B60E16B60DE6E1/zaveri/cell_analysis/datasets/TissueNet_cp_format/platform_specific/{platform}/{tissue}/train'
    #             convert_to_cp_format(d_path, l_path, dest_path)