import lanelines
from compgraph import CompGraph, CompGraphRunner
import numpy as np
import cv2

func_dict = {
    'warp': lanelines.warp,
    'gray': lanelines.gray,
    'get_HLS': lanelines.get_hls_channels,
    'weighted_HLS_sum': lanelines.weighted_HLS,
    'threshold_gray': lanelines.mask_threashold_range,
    'threshold_S': lanelines.mask_threashold_range,
    'threshold_H': lanelines.mask_threashold_range,
    'threshold_wHLS': lanelines.mask_threashold_range,
    'apply_sobel_x_to_S': lanelines.scaled_sobel_x,
    'threshold_S_sobel_x': lanelines.mask_threashold_range,
    'median_blur_tssx': cv2.medianBlur,
    'close_thresholded_S': lanelines.morphological_close,
    'gather_thresholded_images': lanelines.gather_thresholded_images,
    'combine_thresholds_bitwise_or': lanelines.bitwise_or,
    'fit_lane_polynomials': lanelines.fit_lane_polynomials,
    'estimate_curvature': lanelines.lane_curvature,

}

func_io = {
    'warp': (('image', 'M', 'canvas_size'), 'warped'),
    'gray': ('warped', 'warped_gray'),
    'get_HLS': ('warped', ('H', 'L', 'S')),
    'weighted_HLS_sum': (('H', 'L', 'S', 'HLS_weights'), 'weighted_HLS'),
    'threshold_gray': (('warped_gray', 'gray_from', 'gray_to'), 'thresholded_gray'),
    'threshold_S': (('S', 'S_from', 'S_to'), 'thresholded_S'),
    'threshold_H': (('H', 'H_from', 'H_to'), 'thresholded_H'),
    'threshold_wHLS': (('weighted_HLS', 'wHLS_from', 'wHLS_to'), 'thresholded_wHLS'),
    'apply_sobel_x_to_S': ('S', 'S_sobel_x'),
    'threshold_S_sobel_x': (('S_sobel_x', 'S_sobel_x_from', 'S_sobel_x_to'), 'thresholded_S_sobel_x'),
    'median_blur_tssx': (('thresholded_S_sobel_x', 'tssx_median_kernel'), 'tssx_median'),
    'close_thresholded_S': (('thresholded_S', 'close_kernel_for_tS'), 'ts_closed'),
    'gather_thresholded_images' : (
        ('thresholded_S', 'thresholded_wHLS', 'thresholded_S_sobel_x', 'tssx_median', 'ts_closed', 'thresholded_gray'),
        'thresholded_images'
    ),
    'combine_thresholds_bitwise_or': ('thresholded_images', 'all_thresholds'),
    'fit_lane_polynomials': (
        ('all_thresholds', 'n_cells_x', 'n_cells_y', 'cell_threshold'),
        ('p_coefs_left', 'p_coefs_right', 'target_cells_coords_left', 'target_cells_coords_right')
    ),
    'estimate_curvature': (('p_coefs_left', 'p_coefs_right', 'pixels_per_meter', 'canvas_size'), 'curvature')

}

computational_graph = CompGraph(func_dict, func_io)

parameters = {
    'canvas_size': (500, 1500),
    'pixels_per_meter': 113.5,
    'HLS_weights': [0, 0.4, 1.],
    'gray_from': 210,
    'gray_to': 255,
    'S_from': 180,
    'S_to': 255,
    'H_from': 95,
    'H_to': 100,
    'wHLS_from': 180,
    'wHLS_to': 255,
    'S_sobel_x_from': 20,
    'S_sobel_x_to': 240,
    'tssx_median_kernel': 5,
    'close_kernel_for_tS': (3, 3),
    'n_cells_x': 50,
    'n_cells_y': 100,
    'cell_threshold': 70,
}
