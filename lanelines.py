import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def e2h(x):
    return np.array([x[0], x[1], 1.])


def h2e(x):
    x = np.array(x)
    return x[:2] / x[2]


def grayscale(im, flag=cv2.COLOR_BGR2GRAY):
    return cv2.cvtColor(im, flag)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def sobel_x(im):
    return cv2.Sobel(im, cv2.CV_64F, 1, 0)


def sobel_y(im):
    return cv2.Sobel(im, cv2.CV_64F, 0, 1)


def sobel_abs(sobel):
    return scale_image_255(np.abs(sobel))


def sobel_magnitude(sobelx, sobely):
    return np.sqrt(np.square(sobelx) + np.square(sobely))


def sobel_direction(sobelx, sobely):
    return np.arctan2(np.abs(sobely), np.abs(sobelx))


def find_cbc(img, pattern_size, searchwin_size=5):

    findcbc_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS
    res = cv2.findChessboardCorners(img, pattern_size, flags=findcbc_flags)

    found, corners = res
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (searchwin_size, searchwin_size), (-1, -1), term)

    return res


def reformat_corners(corners):
    return corners.reshape(-1, 2)


def get_rectangle_corners_from_cbc(cbc, nx, ny):

    points = np.array([
        cbc[0,:],
        cbc[nx-1,:],
        cbc[-1,:],
        cbc[nx*ny-nx,:],
    ], dtype=np.float32)

    return points


def get_ractangle_corners_in_image(im_sz, offset_x, offset_y):

    points = np.array([
        [offset_x, offset_y],
        [im_sz[0]-offset_x, offset_y],
        [im_sz[0]-offset_x, im_sz[1]-offset_y],
        [offset_x, im_sz[1]-offset_y]
    ], dtype=np.float32)

    return points


def scale_image_255(im):
    return np.uint8(255 * (im / np.max(im)))


def mask_threashold_range(im, thresh_min, thresh_max):

    binary_output = (sb >= thresh_min) & (sb < thresh_max)
    return binary_output


def define_lanes_region(n_rows, n_cols, x_from=450, x_to=518, y_lim=317, left_offset=50, right_offset=0):

    vertices = np.array([[
        [x_from, y_lim],
        [x_to, y_lim],
        [n_cols-right_offset, n_rows],
        [left_offset, n_rows],
    ]], dtype=np.int32)

    return vertices


def apply_region_mask(image, region_vertices):

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, region_vertices, 255)

    return cv2.bitwise_and(image, mask)


def find_hough_lines(im_masked, rho, theta, threshold, min_line_length, max_line_gap):

    lines = cv2.HoughLinesP(im_masked, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines.reshape(lines.shape[0], 4)


def compute_line_tangents(lines):

    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    tans = (y2 - y1) / (x2 - x1)

    return tans


def line_vector_constant_y(val):
    return np.array([0, 1, -val])


def line_vector_from_opencv_points(line):

    x1, y1, x2, y2 = line
    return np.cross([x1, y1, 1], [x2, y2, 1])


def extend_lane_lines(lines, y_const_0, y_const_1):

    n = len(lines)

    res = np.zeros((n, 4), dtype=np.int32)

    line_y0 = line_vector_constant_y(y_const_0)
    line_y1 = line_vector_constant_y(y_const_1)

    for i in range(n):

        line = line_vector_from_opencv_points(lines[i, :])

        intersection_0 = h2e(np.cross(line, line_y0))
        intersection_1 = h2e(np.cross(line, line_y1))

        res[i, :2] = intersection_0
        res[i, 2:] = intersection_1

    return res


def extend_lane_lines_grouped_by_slopes(lines, slopes, y_const_0, y_const_1, abs_slope_threshold=0.2):

    valid_lines = np.abs(slopes) > abs_slope_threshold

    lines_left = extend_lane_lines(lines[np.logical_and(slopes < 0, valid_lines)], y_const_0, y_const_1)
    lines_right = extend_lane_lines(lines[np.logical_and(slopes > 0, valid_lines)], y_const_0, y_const_1)

    return lines_left, lines_right


def average_lines_endpoints(lines):

    return np.array(lines.mean(axis=0), dtype=np.int32)


def lines_distances_to_bottom(lines, n_rows):

    def dist_to_bottom(line):
        y1 = line[1]
        y2 = line[3]
        y_smaller = y1 if y1 < y2 else y2
        return n_rows - y_smaller

    n = len(lines)
    distances = np.zeros(n)

    for i in range(n):
        distances[i] = dist_to_bottom(lines[i, :])

    return distances


def split_distances_to_bottom(distances, slopes):

    return distances[slopes < 0], distances[slopes > 0]


def weighted_average_lines_endpoints(lines, distances_to_bottom):

    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    mu_y1 = y1[0]
    mu_y2 = y2[0]

    weights = 1. / distances_to_bottom
    weights_sum = weights.sum()

    mu_x1 = (x1 * weights).sum() / weights_sum
    mu_x2 = (x2 * weights).sum() / weights_sum

    return np.array([mu_x1, mu_y1, mu_x2, mu_y2], dtype=np.int32)


def weighted_img(im, initial_im, alpha=0.8, beta=1., gamma=0.):
    '''
    dst = initial_im*alpha + im*beta + gamma;
    '''

    return cv2.addWeighted(initial_im, alpha, im, beta, gamma)


def draw_line(canvas_im, line, color=[255, 0, 0], thickness=2):

    x1, y1, x2, y2 = line
    cv2.line(canvas_im, (x1, y1), (x2, y2), color, thickness)


def draw_lines_on_image(canvas_im, lines, color=[255, 0, 0], thickness=2):

    for i in range(lines.shape[0]):
        draw_line(canvas_im, lines[i, :], color, thickness)

def plot_line(line, **kvargs):

    xs = [line[0], line[2]]
    ys = [line[1], line[3]]

    plt.plot(xs, ys, '-', **kvargs)


def plot_homogeneous_line_vector(vec, x_from, x_to, **kvargs):

    a, b, c = vec

    def line_func(x):
        return (-a * x - c) / b

    xs = np.arange(x_from, x_to)
    ys = line_func(xs)

    plt.plot(xs, ys, **kvargs)


def visualize_test_images(images, proc_func=lambda im : im):

    plt.figure(figsize=(16, 16))
    for i, im in enumerate(images):
        plt.subplot(3, 2, i+1)
        plt.imshow(proc_func(im))
