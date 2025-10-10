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

        # Debug: Print what was returned
        print(f"DEBUG: full_names length = {len(full_names)}, full_descriptors length = {len(full_descriptors)}")
        print(f"DEBUG: full_names = {full_names}")
        print(f"DEBUG: full_descriptors = {full_descriptors}")

        # Check if featurization was successful
        if len(full_names) <= 1 and len(full_descriptors) <= 1:
            log_file.write_text(f"FAILED: Insufficient descriptors (names={len(full_names)}, descriptors={len(full_descriptors)})")
            return "FAILED"

        # Success case
        log_file.write_text("SUCCESS")
        return "SUCCESS"

    except (ValueError, NotImplementedError, AssertionError) as e:
        log_file.write_text(f"FAILED: {str(e)}")
        return "FAILED"
    except Exception as e:
        log_file.write_text(f"FAILED: Unexpected error: {str(e)}")
        return "FAILED"


if __name__ == "__main__":
    main()
