import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def open_image(fname, convert_to_rgb=False):

    im = cv2.imread(fname)

    if len(im.shape) == 2:
        return im

    if not convert_to_rgb:
        return im

    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def open_and_undistort_image(fname, cm, dc, convert_to_rgb=False):

    im = open_image(fname, convert_to_rgb)
    return cv2.undistort(im, cm, dc)


def get_im_wh(im):
    h, w = im.shape[:2]
    return w, h


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


def get_rectangle_corners_from_cbc(cbc, nx, ny):

    points = np.array([
        cbc[0,:],
        cbc[nx-1,:],
        cbc[-1,:],
        cbc[nx*ny-nx,:],
    ], dtype=np.float32)

    return points


def get_rectangle_corners_in_image(im_sz, offset_x, offset_y):

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

    binary_output = (im >= thresh_min) & (im < thresh_max)
    return np.uint8(binary_output)


def warp(im, M, canvas_sz):
    return cv2.warpPerspective(im, M, canvas_sz, flags=cv2.INTER_LINEAR)


def convert_to_HLS(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2HLS)


def weighted_sum_images(images, weights):

    assert len(weights) == len(images)

    nonzero_indices = np.nonzero(weights)[0]
    if len(nonzero_indices) < 2:
        raise Exception('At least 2 non-zero weights are required')

    first, second = nonzero_indices[:2]
    res = cv2.addWeighted(images[first], weights[first], images[second], weights[second], 0)

    if len(nonzero_indices) == 2:
        return res

    for i in nonzero_indices[2:]:
        res = cv2.addWeighted(res, 1., images[i], weights[i], 0)

    return res


def bitwise_or(images):

    assert len(images) > 0

    if len(images) == 1:
        return images[0]

    res = cv2.bitwise_or(images[0], images[1])
    if len(images) == 2:
        return res

    for im in images[2:]:
        res = cv2.bitwise_or(res, im)

    return res


def weighted_HLS(H, L, S, weights):
    return weighted_sum_images([H, L, S], weights)


def add_contrast(im, gain):
    gained = gain * im
    return scale_image_255(gained)


def sobel_combo(im):

    sobelx = sobel_x(im)
    sobely = sobel_y(im)

    magnitude = sobel_magnitude(sobelx, sobely)
    direction = sobel_direction(sobelx, sobely)

    return scale_image_255(magnitude), scale_image_255(direction)


def scaled_sobel_x(im):
    return scale_image_255( sobel_x(im) )


def morphological_close(im, kernel=(3, 3)):
    return cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)


def get_hls_channels(im):

    hls = convert_to_HLS(im)

    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]

    return H, L, S

def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def gather_thresholded_images(*images):
    return images

def lane_cells(im, nx, ny, threshold=20):

    cells = divide_image_to_cells(im, nx, ny)

    res = []

    for i in range(ny):

        idx_from = i * nx
        idx_to = i * nx + nx

        rowcells = cells[idx_from:idx_to]

        sums = np.array([np.sum(cell) for cell in rowcells])
        max_j = np.argmax(sums)

        if sums[max_j] > threshold:
            res.append( (i, max_j) )

    return np.array(res)


def lane_cells_real_coords(lanecells, im, nx, ny):

    rows, cols= im.shape[:2]

    cell_sz_x = cols // nx
    cell_sz_y = rows // ny

    points = np.zeros_like(lanecells)

    for i in range(len(lanecells)):
        idx_row, idx_col = lanecells[i, :]
        x = idx_col * cell_sz_x + cell_sz_x / 2
        y = idx_row * cell_sz_y + cell_sz_y / 2
        points[i, :] = (x, y)

    return points


def divide_image_to_cells(im, nx, ny):

    rows, cols= im.shape[:2]

    assert rows % ny == 0
    assert cols % nx == 0

    offset_x = cols // nx
    offset_y = rows // ny

    cells = []

    for j in range(ny):
        for i in range(nx):

            x_from = i * offset_x
            x_to = x_from + offset_x
            y_from = j * offset_y
            y_to = y_from + offset_y

            cell = im[y_from:y_to, x_from:x_to]

            cells.append(cell)

    return cells


def show_cells(cells, nx, ny):

    for i, cell in enumerate(cells):
        plt.subplot(ny, nx, i+1)
        plt.axis('off')
        plt.imshow(cell)


def split_image_lr(im):

    cols = im.shape[1]
    middle = cols // 2
    return im[:, :middle], im[:, middle:]


def split_image_lr_and_show(im):

    left, right = split_image_lr(im)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(left)

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(right)


def get_polynomial_2(coefs):

    a, b, c = coefs

    def f(y):
        return a * (y**2) + b * y + c

    return f

def fit_lane_polynomials(im, nx=50, ny=100, lanecell_threshold=70):

    left, right = split_image_lr(im)

    target_cells_left = lane_cells(left, nx, ny, threshold=70)
    target_cells_coords_left = lane_cells_real_coords(target_cells_left, left, nx, ny)
    p_coefs_left = np.polyfit(target_cells_coords_left[:, 1], target_cells_coords_left[:, 0], 2)

    target_cells_right = lane_cells(right, nx, ny, threshold=70)
    target_cells_coords_right = lane_cells_real_coords(target_cells_right, right, nx, ny)
    target_cells_coords_right[:, 0] += left.shape[1]
    p_coefs_right = np.polyfit(target_cells_coords_right[:, 1], target_cells_coords_right[:, 0], 2)

    return p_coefs_left, p_coefs_right, target_cells_coords_left, target_cells_coords_right



def get_lane_polynomials_points(warped_im, p_coefs_left, p_coefs_right):

    poly_left = get_polynomial_2(p_coefs_left)
    poly_right = get_polynomial_2(p_coefs_right)

    poly_y = np.linspace(0, warped_im.shape[0])
    poly_x_left = poly_left(poly_y)
    poly_x_right = poly_right(poly_y)

    return poly_y, poly_x_left, poly_x_right


def lanefill(image, warped, Minv, poly_y, poly_x_left, poly_x_right):

    canvas = np.zeros_like(warped).astype(np.uint8)

    pts_left = np.array([np.transpose(np.vstack([poly_x_left, poly_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([poly_x_right, poly_y])))])
    pts = np.hstack((pts_left, pts_right)).astype(np.int32)

    cv2.fillPoly(canvas, [pts], (0, 255, 0))

    newwarp = cv2.warpPerspective(canvas, Minv, (image.shape[1], image.shape[0]))

    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result


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


def move_line(line, offset_x=0., offset_y=0.):

    x1, y1, x2, y2 = line

    return np.array([
        x1 + offset_x,
        y1 + offset_y,
        x2 + offset_x,
        y2 + offset_y
    ])


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


def imshow_bgr(im, axis_setting='off'):

    plt.axis(axis_setting)
    plt.imshow( cv2.cvtColor(im, cv2.COLOR_BGR2RGB) )
