"""RAC_finder.py
Compute revised autocorrelation functions (RACs) for feature extraction.

Optimized I/O pipeline. Original design by Kulik's Group @MIT:
- https://github.com/hjkgrp/molSimplify
- https://github.com/hjkgrp/MOFSimplify
"""

import os
import sys
from pathlib import Path

from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors


def main():
    """Extract MOF descriptors and write results to log file."""
    if len(sys.argv) < 3:
        print("Usage: python RAC_finder.py <primitive_path> <RAC_dir>")
        sys.exit(1)

    primitive_path = sys.argv[1]
    rac_dir = Path(sys.argv[2])
    log_file = rac_dir / "RAC_getter_log.txt"

    try:
        full_names, full_descriptors = get_MOF_descriptors(
            primitive_path,
            3,
            path=str(rac_dir),
            xyzpath=str(rac_dir / "temp.xyz"),
        )

        # Check if featurization was successful
        if len(full_names) <= 1 and len(full_descriptors) <= 1:
            log_file.write_text("FAILED")
            return "FAILED"

    except (ValueError, NotImplementedError, AssertionError):
        log_file.write_text("FAILED")
        return "FAILED"


if __name__ == "__main__":
    main()
