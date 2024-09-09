"""extract_all.py
Run feature_extraction commands on all CIF files in the specified directory. 
"""

import os
import sys
import pandas as pd
from mpi4py import MPI
import multiprocessing as mp
from feature_extraction import extract_features


def _mpi_extract(cif_file, src_dir, project_path):
    """
    Extract features from a given cif file. This function runs in an MPI environment
    and involves file I/O through extract_features.

    :param cif_file: the name of the target cif file for feature extraction
    :param src_dir: source directory for cif files
    :param project_path: path to the project
    """
    target_path = os.path.join(src_dir, cif_file)
    cif_without_suffix = cif_file[:-4]
    cur_feature_df = extract_features(project_path, target_path, id=cif_without_suffix)
    return cur_feature_df


def extract_all(project_path, src_dir, dest_dir, max_lim=20):
    """
    Extract features from all cif files in the src_dir using MPI for parallel processing.

    :param project_path: path to the project
    :param src_dir: target directory for feature extraction
    :param dest_dir: destination directory for the results
    :param max_lim: maximum number of cif files to process, default to 20
    """
    if not os.path.isdir(src_dir):
        print(f"Invalid source directory: {src_dir}")
        return
    if not os.path.isdir(dest_dir):
        print(f"Invalid destination directory: {dest_dir}")
        return

    # MPI Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Root process
    if rank == 0:
        cif_files = [file for file in os.listdir(src_dir) if file.endswith(".cif")]
        cif_files = cif_files[:max_lim]
    else:
        cif_files = None

    # Broadcast the list of cif files to all processes
    cif_files = comm.bcast(cif_files, root=0)

    # Scatter the files among processes
    local_files = cif_files[rank::size]

    # Each process extracts features for its assigned files
    local_results = []
    for cif_file in local_files:
        local_results.append(_mpi_extract(cif_file, src_dir, project_path))

    # Gather the results at the root process
    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        all_feature_df = pd.concat(
            [df for sublist in gathered_results for df in sublist]
        )
        return all_feature_df


def extract_all_to_csv(
    project_path, src_dir, dest_dir, file_name="features.csv", max_lim=20
) -> None:
    """
    Extract features from all cif files in the src_dir

    :param src_dir: target directory for feature extraction
    :param dest_dir: destination directory for the results
    :param max_lim: maximum number of cif files to process, default to 100
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

    all_feature_df = extract_all(project_path, src_dir, dest_dir, max_lim)
    all_feature_df.to_csv(export_file_path)


if __name__ == "__main__":
    print("***Running extract_all as main***")
    project_path = "."
    src_dir = os.path.join(".", "CoRE2019")
    dest_dir = os.path.join(".", "data")
    extract_all_to_csv(project_path, src_dir, dest_dir, max_lim=200)
    print("***extract_all exited successfully***")
