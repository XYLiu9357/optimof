"""extract_all.py
Run feature_extraction commands on all CIF files in the specified directory. 
"""

import os
import sys
import pandas as pd
import multiprocessing as mp
from feature_extraction import extract_features


def _mp_extract(cif_file):
    """
    Extract features from a given cif file. This function runs in a multiprocessing
    environment and involves file I/O through extract_features.

    :param cif_file: the name of the target cif file for feature extraction
    """

    target_path = os.path.join(src_dir, cif_file)
    cif_without_suffix = cif_file[:-4]
    cur_feature_df = extract_features(project_path, target_path, id=cif_without_suffix)
    return cur_feature_df


def extract_all(src_dir, max_lim=20) -> pd.DataFrame:
    """
    Extract features from all cif files in the src_dir

    :param src_dir: target directory for feature extraction
    :param lim: maximum number of cif files to process, default to 100
    """
    if not os.path.isdir(src_dir):
        print(f"Invalid source directory: {src_dir}")
    if not os.path.isdir(dest_dir):
        print(f"Invalid destination directory: {dest_dir}")

    cif_files = [file for file in os.listdir(src_dir) if file.endswith(".cif")]
    cif_files = cif_files[: min(max_lim, len(cif_files))]

    # Create as many processes as we can
    num_process = mp.cpu_count()
    with mp.Pool(processes=min(len(cif_files), num_process)) as pool:
        results = pool.map(_mp_extract, cif_files)

    all_feature_df = pd.concat(results)
    return all_feature_df


def extract_all_to_csv(src_dir, dest_dir, file_name="features.csv", max_lim=20) -> None:
    """
    Extract features from all cif files in the src_dir

    :param src_dir: target directory for feature extraction
    :param lim: maximum number of cif files to process, default to 100
    """
    if not os.path.isdir(src_dir):
        print(f"Invalid source: {src_dir}")
    if not os.path.isdir(dest_dir):
        print(f"Invalid destination: {dest_dir}")

    # Export path
    export_file_path = os.path.join(dest_dir, file_name)

    # Overwrite warning
    if any(file_name in file_found for file_found in os.listdir(dest_dir)):
        do_overwrite = input(
            f"WARNING: {file_name} found in destination. Overwrite? [y/n] "
        )
        while True:
            if do_overwrite == "y":
                print("Feature extraction proceeds...")
                os.remove(export_file_path)
                break
            elif do_overwrite == "n":
                print("Feature extraction terminated by user...")
                return
            else:
                print(f"Invalid option {do_overwrite}")

    all_feature_df = extract_all(src_dir, max_lim)
    all_feature_df.to_csv(export_file_path)


if __name__ == "__main__":
    print("***Running extract_all as main***")
    project_path = "."
    src_dir = os.path.join(".", "CoRE2019")
    dest_dir = os.path.join(".", "data")
    extract_all_to_csv(src_dir, dest_dir)
    print("***extract_all exited successfully***")
