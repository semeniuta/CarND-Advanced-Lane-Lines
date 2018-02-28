import cv2
import numpy as np
import os
from glob import glob


def get_image_size(im):

    h, w = im.shape
    return (w, h)


def find_cbc(im, pattern_size, searchwin_size=5):

    findcbc_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS
    res = cv2.findChessboardCorners(im, pattern_size, flags=findcbc_flags)

    found, corners = res
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(im, corners, (searchwin_size, searchwin_size), (-1, -1), term)

    return res


def reformat_corners(corners):
    return corners.reshape(-1, 2)


def calibrate_camera(image_points, im_sz, pattern_size, square_size):

    n_images = len(image_points)

    object_points = get_object_points(n_images, pattern_size, square_size)

    return cv2.calibrateCamera(object_points, image_points, im_sz, None, None)


def get_object_points(num_images, pattern_size, square_size):

    pattern_points = get_pattern_points(pattern_size, square_size)
    object_points = [pattern_points for i in range(num_images)]
    return object_points


def get_pattern_points(pattern_size, square_size):

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    return pattern_points


def do_calibration(im_dir, psize, sqsize):

    mask = im_dir + '/*.jpg'
    imfiles = glob(mask)

    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in imfiles]
    cbc = (find_cbc(im, psize) for im in images)

    image_points = []
    for found, corners in cbc:
        if found:
            image_points.append(reformat_corners(corners))

    print('Successful corner detection in {:d}/{:d} images'.format(len(image_points), len(images)))

    imsize = get_image_size(images[0])
    rms, cm, dc, rvecs, tvecs = calibrate_camera(image_points, imsize, psize, sqsize)

    print('Camera matrix:')
    print(cm)

    print('Distortion coefficients:')
    print(dc)

    return cm, dc


if __name__ == '__main__':

    cm, dc = do_calibration('camera_cal', psize=(9, 6), sqsize=1.)

    np.save('camera_cal_results/camera_matrix.npy', cm)
    np.save('camera_cal_results/dist_coefs.npy', dc)
