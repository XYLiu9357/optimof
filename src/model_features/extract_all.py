"""extract_all.py
Run feature_extraction commands on all CIF files in the specified directory.
Includes progress tracking and error handling for batch processing.
"""

import multiprocessing as mp
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .feature_extraction import extract_features

# Global variables for multiprocessing (set by extract_all)
_project_path = None
_src_dir = None


def _mp_extract(cif_file: str) -> pd.DataFrame:
    """Extract features from a given cif file in multiprocessing environment.

    This function runs in a separate process and performs I/O through extract_features.

    Args:
        cif_file: Name of the target CIF file for feature extraction

    Returns:
        DataFrame with extracted features, or empty DataFrame if extraction fails
    """
    target_path = _src_dir / cif_file
    cif_without_suffix = cif_file[:-4] if cif_file.endswith(".cif") else cif_file

    try:
        result_df = extract_features(
            _project_path, str(target_path), id=cif_without_suffix
        )
        if result_df.empty:
            print(f"✗ Failed to extract features from {cif_file}")
            return pd.DataFrame()
        return result_df
    except Exception as e:
        print(f"✗ Error processing {cif_file}: {e}")
        return pd.DataFrame()


def extract_all(
    src_dir: Path,
    project_path: Path = Path("."),
    max_lim: Optional[int] = None,
) -> pd.DataFrame:
    """Extract features from all CIF files in the source directory.

    Args:
        src_dir: Target directory containing CIF files
        project_path: Root path of the project (default: current directory)
        max_lim: Maximum number of CIF files to process (None = all files)

    Returns:
        Combined DataFrame with all extracted features

    Raises:
        ValueError: If source directory doesn't exist or contains no CIF files
    """
    # Set globals for multiprocessing
    global _project_path, _src_dir
    _project_path = Path(project_path)
    _src_dir = Path(src_dir)

    # Validate inputs
    if not _src_dir.is_dir():
        raise ValueError(f"Invalid source directory: {_src_dir}")

    # Find all CIF files
    cif_files = sorted([file.name for file in _src_dir.glob("*.cif")])
    if not cif_files:
        raise ValueError(f"No CIF files found in {_src_dir}")

    # Apply limit if specified
    if max_lim is not None:
        cif_files = cif_files[:max_lim]

    print(f"Found {len(cif_files)} CIF files to process")
    print(f"Using {mp.cpu_count()} CPU cores")

    # Create multiprocessing pool
    num_processes = min(len(cif_files), mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(_mp_extract, cif_files)

    # Filter out empty results (failed extractions)
    valid_results = [df for df in results if not df.empty]

    if not valid_results:
        raise RuntimeError("All feature extractions failed")

    print(
        f"Successfully extracted features from {len(valid_results)}/{len(cif_files)} files"
    )

    # Combine all results
    all_feature_df = pd.concat(valid_results, ignore_index=True)
    return all_feature_df


def extract_all_to_csv(
    src_dir: Path,
    dest_dir: Path,
    file_name: str = "features.csv",
    project_path: Path = Path("."),
    max_lim: Optional[int] = None,
) -> None:
    """Extract features from all CIF files and save to CSV.

    Args:
        src_dir: Target directory containing CIF files
        dest_dir: Destination directory for output CSV
        file_name: Output filename (default: "features.csv")
        project_path: Root path of the project
        max_lim: Maximum number of CIF files to process (None = all files)

    Raises:
        ValueError: If source or destination directory doesn't exist
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)

    # Validate paths
    if not src_dir.is_dir():
        raise ValueError(f"Invalid source directory: {src_dir}")
    if not dest_dir.is_dir():
        raise ValueError(f"Invalid destination directory: {dest_dir}")

    export_file_path = dest_dir / file_name

    # Check if output file already exists
    if export_file_path.exists():
        response = input(
            f"WARNING: {file_name} exists in {dest_dir}. Overwrite? [y/n] "
        )
        if response.lower() != "y":
            print("Feature extraction cancelled by user")
            return
        export_file_path.unlink()

    # Extract features
    print(f"\nExtracting features from {src_dir}")
    all_feature_df = extract_all(src_dir, project_path, max_lim)

    # Save to CSV
    all_feature_df.to_csv(export_file_path, index=False)
    print(f"\n✓ Saved {all_feature_df.shape} to {export_file_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("BATCH FEATURE EXTRACTION")
    print("=" * 70)

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python extract_all.py <cif_directory> [max_files]")
        print("\nExample:")
        print("  python extract_all.py CoRE2019/ 100")
        sys.exit(1)

    project_path = Path(".")
    src_dir = Path(sys.argv[1])
    max_lim = int(sys.argv[2]) if len(sys.argv) > 2 else None

    dest_dir = project_path / "data"

    try:
        extract_all_to_csv(
            src_dir=src_dir,
            dest_dir=dest_dir,
            file_name="features.csv",
            project_path=project_path,
            max_lim=max_lim,
        )
        print("\n" + "=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)
