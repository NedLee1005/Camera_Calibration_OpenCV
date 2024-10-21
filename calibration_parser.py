import argparse
from array import array
def calibration_parser():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--size", help="Enter number of row/col points ex:6 7", nargs = 2, type=int, default=[6, 8])
    parser.add_argument(
        "-m", "--mm", help="enter square length of chessboard defaute is 1", type=int, default=1)
    parser.add_argument(
        "-S", "--save_dir", help="directory to save undistorted images", default="undistorted_image")
    parser.add_argument(
        "-l", "--load_dir", help="directory that contain images for camera calibration", default="test")
    return parser