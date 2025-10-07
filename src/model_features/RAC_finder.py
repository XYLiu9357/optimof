"""RAC_finder.py
Compute revised autocorrelation functions (RACs) for feature extraction.

Optimized I/O pipeline. Original design by Kulik's Group @MIT:
- https://github.com/hjkgrp/molSimplify
- https://github.com/hjkgrp/MOFSimplify
"""

import os
import sys

from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors


def main():

    # user command line inputs
    primitive_path = sys.argv[1]  # name of the MOF
    RAC_dir = sys.argv[2]

    print(RAC_dir)

    # result log
    f = open(os.path.join(RAC_dir, "RAC_getter_log.txt"), "w")

    try:
        # makes the linkers and sbus folders
        full_names, full_descriptors = get_MOF_descriptors(
            primitive_path,
            3,
            path=RAC_dir,
            xyzpath=os.path.join(RAC_dir, "temp.xyz"),
        )
    except ValueError:
        f.write("FAILED")
        f.close()
        return "FAILED"
    except NotImplementedError:
        f.write("FAILED")
        f.close()
        return "FAILED"
    except AssertionError:
        f.write("FAILED")
        f.close()
        return "FAILED"

    if (len(full_names) <= 1) and (
        len(full_descriptors) <= 1
    ):  # this is a featurization check from MOF_descriptors.py
        f.write("FAILED")
        f.close()
        return "FAILED"


if __name__ == "__main__":
    main()
