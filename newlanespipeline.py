import lanelines
from compgraph import CompGraph, CompGraphRunner
import numpy as np
import cv2

func_dict = {
    'get_image_shape': lambda im : im.shape,
    'define_lanes_region': lanelines.define_lanes_region,
    'apply_region_mask': lanelines.apply_region_mask
}

func_io = {
    'get_image_shape': ('image', ('n_rows', 'n_cols')),
    'define_lanes_region': (
        ('n_rows', 'n_cols', 'x_from', 'x_to', 'y_lim', 'left_offset', 'right_offset'),
        'region_vertices'
    ),
    'apply_region_mask': (('image', 'region_vertices'), 'masked_image'),
}

computational_graph = CompGraph(func_dict, func_io)

parameters = {
    'x_from': 560,
    'x_to': 710,
    'y_lim': 450,
    'left_offset': 50,
    'right_offset': 0,
}
