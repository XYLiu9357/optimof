"""extract_all.py
Run feature_extraction commands on all CIF files in the specified directory. 
"""

import os
import sys
import pandas as pd
import multiprocessing as mp
from feature_extraction import extract_features

def _mp_extract(cif_file): 
    target_path = os.path.join(src_dir, cif_file)
    cif_without_suffix = cif_file[:-4] 
    cur_feature_df = extract_features(project_path, target_path, id=cif_without_suffix)
    return cur_feature_df

def extract_all(src_dir) -> pd.DataFrame: 
    """
    Extract features from all cif files in the src_dir
    """
    if not os.path.isdir(src_dir): 
        print(f"Invalid source directory: {src_dir}")
    if not os.path.isdir(dest_dir): 
        print(f"Invalid destination directory: {dest_dir}")
    
    cif_files = [file for file in os.listdir(src_dir) if file.endswith('.cif')]
    cif_files = cif_files[:4]
    
    # Create as many processes as we can
    num_process = mp.cpu_count()
    with mp.Pool(processes=max(len(cif_files), num_process)) as pool: 
        results = pool.map(_mp_extract, cif_files)
    
    all_feature_df = pd.concat(results, axis=1)
    return all_feature_df

def extract_all_to_csv(src_dir, dest_dir, file_name="features.csv") -> None: 
    if not os.path.isdir(src_dir): 
        print(f"Invalid source: {src_dir}")
    if not os.path.isdir(dest_dir): 
        print(f"Invalid destination: {dest_dir}")

    # Export path
    export_file_path = os.path.join(dest_dir, file_name)

    # Overwrite warning 
    if any(file_name in file_found for file_found in os.listdir(dest_dir)): 
        do_overwrite = input(f"WARNING: {file_name} found in destination. Overwrite? [y/n] ")
        while True: 
            if (do_overwrite == "y"): 
                print("Feature extraction proceeds...")
                os.remove(export_file_path)
                break
            elif (do_overwrite == "n"): 
                print("Feature extraction terminated by user...")
                return
            else: 
                print(f"Invalid option {do_overwrite}")

    all_feature_df = extract_all(src_dir)
    all_feature_df.to_csv(export_file_path)

if __name__ == "__main__": 
    print("***Running extract_all as main***")
    project_path = "."
    src_dir = os.path.join(".", "CoRE2019")
    dest_dir = os.path.join(".", "test")
    extract_all_to_csv(src_dir, dest_dir)
    print("***extract_all exited successfully***")
