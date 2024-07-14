"""descriptor.py
Generate integrated descriptors from .cif files using Zeo++ library
and RAC_finder module. 

Modified I/O pipeline based on design by Kulik's Group @MIT: 
- https://github.com/hjkgrp/molSimplify
- https://github.com/hjkgrp/MOFSimplify

Important Notes: 
- Only supports openbabel 2.4.1. Newer versions will report error
"""

import os
import shutil
import subprocess
import numpy as np
import pandas as pd

def generate_descriptor(project_path, name, structure, prediction_type, is_entry):
    """
    Inputs are the name of the MOF and the structure (cif file text) of the MOF for which descriptors are to be generated.
    The third input indicates the type of prediction (solvent removal or thermal).

    :param project_path: str, root path of the predictor project
    :param name: str, the name of the MOF being analyzed
    :param structure: str, the text of the cif file of the MOF being analyzed.
    :param prediction_type: str, the type of prediction being run. Can either be 'solvent' or 'thermal'
    :param is_entry: boolean, indicates whether the descriptor CSV has already been written

    :return: str or dict or array
        1. The string 'FAILED' (if descriptor generation fails)
        2. The string myResult (directory with the generated features)
    """

    # Reinitialize: remove existing temp directory
    if "temp" in os.listdir(project_path):
        shutil.rmtree(os.path.join(project_path, "temp"))
    os.mkdir(os.path.join(project_path, "temp"))

    temp_dir = project_path + "temp/"
    cif_dir = temp_dir  # Subjected to change

    # Write data back
    try:
        cif_file = open(cif_dir + name + ".cif", "w")
    except FileNotFoundError:
        return "FAILED"
    cif_file.write(structure)
    cif_file.close()

    # Construct temp folder structure
    feature_dir = os.path.join(temp_dir + "feature_generation")
    RAC_dir = os.path.join(feature_dir, prediction_type + "_RAC")
    zeo_dir = os.path.join(feature_dir, prediction_type + "_zeo++")
    os.mkdir(feature_dir)
    os.mkdir(RAC_dir)
    os.mkdir(zeo_dir)

    if not is_entry:  # have to generate the CSV

        # Next, running MOF featurization
        shutil.copy(cif_dir + name + ".cif", cif_dir + name + "_primitive.cif")

        # get_MOF_descriptors is used in RAC_getter.py to get RAC features.
        # The files that are generated from RAC_getter.py: lc_descriptors.csv, sbu_descriptors.csv, linker_descriptors.csv

        # cmd1, cmd2, and cmd3 are for Zeo++. cmd4 is for RACs.
        cmd1 = (
            project_path
            + "zeo++-0.3/network -ha -res "
            + zeo_dir
            + name
            + "_pd.txt "
            + cif_dir
            + name
            + "_primitive.cif"
        )
        cmd2 = (
            project_path
            + "zeo++-0.3/network -sa 1.86 1.86 10000 "
            + zeo_dir
            + name
            + "_sa.txt "
            + cif_dir
            + name
            + "_primitive.cif"
        )
        cmd3 = (
            project_path
            + "zeo++-0.3/network -volpo 1.86 1.86 10000 "
            + zeo_dir
            + name
            + "_pov.txt "
            + cif_dir
            + name
            + "_primitive.cif"
        )
        cmd4 = (
            "python "
            + project_path
            + "src/RAC_finder.py %s %s %s" % (cif_dir, name, RAC_dir)
        )

        # four parallelized Zeo++ and RAC commands
        process1 = subprocess.Popen(
            cmd1, stdout=subprocess.PIPE, stderr=None, shell=True
        )
        process2 = subprocess.Popen(
            cmd2, stdout=subprocess.PIPE, stderr=None, shell=True
        )
        process3 = subprocess.Popen(
            cmd3, stdout=subprocess.PIPE, stderr=None, shell=True
        )
        process4 = subprocess.Popen(
            cmd4, stdout=subprocess.PIPE, stderr=None, shell=True
        )

        output1 = process1.communicate()[0]
        output2 = process2.communicate()[0]
        output3 = process3.communicate()[0]
        output4 = process4.communicate()[0]

        # Have written output of Zeo++ commands to files. Now, code below extracts information from those files.

        """ 
        The geometric descriptors are:
        - the maximum included sphere (Di)
        - maximum free sphere (Df)
        - maximum included sphere in the free sphere path (Dif)
        - gravimetric pore volume (GPOV)
        - volumetric pore volume (VPOV)
        - gravimetric surface area (GSA)
        - volumetric surface area (VSA)
        - cell volume (cell_v)
        - gravimetric pore accessible volume (GPOAV)
        - gravimetric pore non-accessible volume (GPONAV)
        - pore-accessible volume (POAV)
        - pore non-accessible volume (PONAV)
        - pore accessible void fraction (POAVF)
        - pore nonaccessible void fraction (PONAVF)

        All Zeo++ calculations use a probe radius of 1.86 angstrom, and zeo++ is called by subprocess.
        """

        dict_list = []
        cif_file = name + "_primitive.cif"
        basename = cif_file.strip(".cif")
        (
            largest_included_sphere,
            largest_free_sphere,
            largest_included_sphere_along_free_sphere_path,
        ) = (np.nan, np.nan, np.nan)
        unit_cell_volume, crystal_density, VSA, GSA = np.nan, np.nan, np.nan, np.nan
        VPOV, GPOV = np.nan, np.nan
        POAV, PONAV, GPOAV, GPONAV, POAV_volume_fraction, PONAV_volume_fraction = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

        if (
            os.path.exists(zeo_dir + name + "_pd.txt")
            & os.path.exists(zeo_dir + name + "_sa.txt")
            & os.path.exists(zeo_dir + name + "_pov.txt")
        ):
            with open(zeo_dir + name + "_pd.txt") as f:
                pore_diameter_data = f.readlines()
                for row in pore_diameter_data:
                    largest_included_sphere = float(
                        row.split()[1]
                    )  # largest included sphere
                    largest_free_sphere = float(row.split()[2])  # largest free sphere
                    largest_included_sphere_along_free_sphere_path = float(
                        row.split()[3]
                    )  # largest included sphere along free sphere path
            with open(zeo_dir + name + "_sa.txt") as f:
                surface_area_data = f.readlines()
                for i, row in enumerate(surface_area_data):
                    if i == 0:
                        unit_cell_volume = float(
                            row.split("Unitcell_volume:")[1].split()[0]
                        )  # unit cell volume
                        crystal_density = float(
                            row.split("Density:")[1].split()[0]
                        )  # crystal density
                        VSA = float(
                            row.split("ASA_m^2/cm^3:")[1].split()[0]
                        )  # volumetric surface area
                        GSA = float(
                            row.split("ASA_m^2/g:")[1].split()[0]
                        )  # gravimetric surface area
            with open(zeo_dir + name + "_pov.txt") as f:
                pore_volume_data = f.readlines()
                for i, row in enumerate(pore_volume_data):
                    if i == 0:
                        density = float(row.split("Density:")[1].split()[0])
                        POAV = float(
                            row.split("POAV_A^3:")[1].split()[0]
                        )  # Probe accessible pore volume
                        PONAV = float(
                            row.split("PONAV_A^3:")[1].split()[0]
                        )  # Probe non-accessible probe volume
                        GPOAV = float(row.split("POAV_cm^3/g:")[1].split()[0])
                        GPONAV = float(row.split("PONAV_cm^3/g:")[1].split()[0])
                        POAV_volume_fraction = float(
                            row.split("POAV_Volume_fraction:")[1].split()[0]
                        )  # probe accessible volume fraction
                        PONAV_volume_fraction = float(
                            row.split("PONAV_Volume_fraction:")[1].split()[0]
                        )  # probe non accessible volume fraction
                        VPOV = POAV_volume_fraction + PONAV_volume_fraction
                        GPOV = VPOV / density
        else:
            print(
                "Not all 3 files exist, so at least one Zeo++ call failed!",
                "sa: ",
                os.path.exists(zeo_dir + name + "_sa.txt"),
                "; pd: ",
                os.path.exists(zeo_dir + name + "_pd.txt"),
                "; pov: ",
                os.path.exists(zeo_dir + name + "_pov.txt"),
            )
            return "FAILED"
        geo_dict = {
            "name": basename,
            "cif_file": cif_file,
            "Di": largest_included_sphere,
            "Df": largest_free_sphere,
            "Dif": largest_included_sphere_along_free_sphere_path,
            "cell_v": unit_cell_volume,
            "VSA": VSA,
            "GSA": GSA,
            "VPOV": VPOV,
            "GPOV": GPOV,
            "POAV_vol_frac": POAV_volume_fraction,
            "PONAV_vol_frac": PONAV_volume_fraction,
            "GPOAV": GPOAV,
            "GPONAV": GPONAV,
            "POAV": POAV,
            "PONAV": PONAV,
        }
        dict_list.append(geo_dict)
        geo_df = pd.DataFrame(dict_list)
        geo_df.to_csv(zeo_dir + "geometric_parameters.csv", index=False)

        # error handling for cmd4
        with open(RAC_dir + "RAC_getter_log.txt", "r") as f:
            if f.readline() == "FAILED":
                print("RAC generation failed.")
                return "FAILED"
                
        # Merging geometric information with the RAC information that is in the get_MOF_descriptors-generated files (lc_descriptors.csv, sbu_descriptors.csv, linker_descriptors.csv)
        try:
            lc_df = pd.read_csv(os.path.join(RAC_dir, "lc_descriptors.csv"))
            sbu_df = pd.read_csv(os.path.join(RAC_dir, "sbu_descriptors.csv"))
            linker_df = pd.read_csv(os.path.join(RAC_dir, "linker_descriptors.csv"))
        except Exception:  # csv files have been deleted
            return "FAILED"

        lc_df_numeric_mean = lc_df.select_dtypes(include=[np.number]).mean().to_frame().transpose()
        sbu_df_numeric_mean = sbu_df.select_dtypes(include=[np.number]).mean().to_frame().transpose()
        linker_df_numeric_mean = linker_df.select_dtypes(include=[np.number]).mean().to_frame().transpose()

        merged_df = pd.concat([geo_df, lc_df_numeric_mean, sbu_df_numeric_mean, linker_df_numeric_mean], axis=1)

        merged_dir = temp_dir + "merged_descriptors"
        if "merged_descriptors" in os.listdir(temp_dir): 
            os.rmdir(merged_dir)
        os.mkdir(merged_dir)

        merged_df.to_csv(
            os.path.join(temp_dir, "merged_descriptors", name + "_descriptors.csv"),
            index=False,
        )  # written in /temp_file_creation_SESSIONID

    else:  # CSV is already written
        merged_df = pd.read_csv(
            os.path.join(temp_dir, "merged_descriptors", name + "_descriptors.csv"),
        )

    myResult = temp_dir
    return myResult


# Generate descriptors based on a test file
if __name__ == "__main__":

    # Initializer
    project_path = os.path.abspath(".") + "/"
    target = "test/ABAVIJ_clean.cif"
    shutil.copyfile(target, "temp/test.cif")

    # Read structure
    structure = "FAILED"
    with open(target, "r") as test_cif:
        structure = test_cif.read()  # Potential buffer overflow
    
    # Assumes under project_path, there is a temp directory containing 
    # the specified cif file. 

    # Generate descriptors 
    print("Running descriptor generator as main")
    result = generate_descriptor(project_path, "test", structure, "thermal", False)

    if result == "FAILED": 
        print("generate_descriptors exit due to failure")
        exit(1)
    else: 
        print(f"generate_descriptors exit successfully. Results stored at {result}")
    
    # Print out features obtained 
    print("### Print features obtained ###")
    feature_path = os.path.join(result, "merged_descriptors", "test_descriptors.csv")
    feature_df = pd.read_csv(feature_path)
    print(feature_df)

