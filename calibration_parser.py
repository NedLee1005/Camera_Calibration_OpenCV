import argparse
from array import array
def calibration_parser():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--size", help="Enter number of row/col points ex:6 7", type=array, default=[6, 7])
    parser.add_argument(
        "-m", "--mm", help="enter square length of chessboard defaute is 1", type=int, default=1)
    parser.add_argument(
        "-S", "--save_dir", help="directory to save undistorted images", default="CalibrateImage")
    parser.add_argument(
        "-l", "--load_dir", help="directory that contain images for camera calibration", default="Test")
    return parser