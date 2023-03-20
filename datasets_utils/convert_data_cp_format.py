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
    for split in ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]:
        d_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_split_cleaned/{split}/test/data'
        l_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_split_cleaned/{split}/test/labels'
        dest_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_Split_Cellpose_format/{split}/test'
        convert_to_cp_format(d_path, l_path, dest_path)

        d_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_split_cleaned/{split}/train/data'
        l_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_split_cleaned/{split}/train/labels'
        dest_path = f'/mnt/5400C9CC66E778B9/Ram/work/cell_analysis/datasets/datasets/LiveCell/livecell_split/Livecell_Split_Cellpose_format/{split}/train'
        convert_to_cp_format(d_path, l_path, dest_path)