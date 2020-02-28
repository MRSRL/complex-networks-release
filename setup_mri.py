"""Script for setup"""
import argparse
import sys

import mri_prep

def main(argv):
    """Parse args and executes commands."""
    parser = argparse.ArgumentParser(description="Setup dataset for training")
    parser.add_argument(
        "-d", "--download", default="raw", help="download directory (default: raw)"
    )
    parser.add_argument(
        "-m", "--masks", default="masks", help="mask directory (default: mask)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose printing")
    parser.add_argument(
        "-o", "--output", default="data", help="final data directory (default: data)"
    )
    args = parser.parse_args()

    verbose = args.verbose
    dir_download = args.download
    dir_masks = args.masks
    dir_output = args.output

        if verbose:
            print("<<< Downloading data... >>>")
        mri_prep.download_dataset_knee(dir_download, verbose=verbose)

        if verbose:
            print("<<< Creating masks... >>>")
        mri_prep.create_masks(dir_masks, verbose=verbose)

    if verbose:
        print("<<< Preparing TFRecords...>>>")
    mri_prep.setup_data_tfrecords(dir_download, dir_output, verbose=verbose)

    if verbose:
        print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])