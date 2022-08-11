
import numpy as np
import cv2 as cv
import glob
import os
from calibration_parser import calibration_parser

dir_path = os.path.dirname(os.path.abspath(__file__))


def check_camera_matrix_exist():
    return os.path.exists(dir_path+"/cam_matrix.npy")


def calibrate_camera(row=6, col=7, mm=1, dataPath="Test"):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....
    objp = np.zeros((row*col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col*mm:mm, 0:row*mm:mm].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # retrive images from Test
    images = glob.glob('{}/*.jpg'.format(dataPath))
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (col, row), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            #  more precise
            cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    mtx, dist = check_reproject_error(objpoints, imgpoints, gray)

    # store the camera matrix and distortion coefficients
    np.save('cam_matrix', mtx)
    np.save('cam_distortion', dist)
    print("matrix save !!")


def check_reproject_error(objpoints, imgpoints, gray):
    mean_error = 0
    #  camera matrix, distortion coefficients, rotation and translation vectors
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2,
                            cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print("total error: {}".format(mean_error/len(objpoints)))
    return mtx, dist


def undistort_image(images, save_dir):
    mtx = np.load('cam_matrix.npy')
    dist = np.load('cam_distortion.npy')
    print("===camera matrix===\n",mtx,"\n")
    print("===distortion coefficients===\n",dist,"\n")
    print("----------transform images---------------")
    # undistortion
    for fname in images:
        img = cv.imread(fname)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))

        # use cv.undistort()
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        fname = os.path.basename(fname)
        cv.imwrite("{}/{}".format(save_dir, fname), dst)
    print("save Done!")


if __name__ == '__main__':
    # args
    parser = calibration_parser()
    args = parser.parse_args()

    if not check_camera_matrix_exist():
        calibrate_camera(args.size[0], args.size[1], args.mm, args.load_dir)
    images = glob.glob('Test/*.jpg')
    undistort_image(images, args.save_dir)
