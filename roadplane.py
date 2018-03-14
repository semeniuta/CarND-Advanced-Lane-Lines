import cv2
import numpy as np

import lanelines
import straightlanespipeline as slp
from compgraph import CompGraphRunner


def define_flat_plane_on_road(images, x_offset=0):

    runner = CompGraphRunner(
        slp.computational_graph,
        frozen_tokens=slp.parameters
    )

    left_lines = []
    right_lines = []
    for im in images:
        runner.run(image=im)
        left = lanelines.move_line(runner['avg_line_left'], -x_offset)
        right = lanelines.move_line(runner['avg_line_right'], x_offset)

        left_lines.append(left)
        right_lines.append(right)

    avg_left = np.array(left_lines).mean(axis=0)
    avg_right = np.array(right_lines).mean(axis=0)

    res = np.array([
        avg_left[:2],
        avg_right[:2],
        avg_right[2:],
        avg_left[2:],
    ], dtype=np.float32)

    return res


def prepare_perspective_transforms(straight_images, canvas_sz, offset_x, offset_y):

    warp_src = define_flat_plane_on_road(
        straight_images,
        x_offset=0
    )

    warp_dst = lanelines.get_rectangle_corners_in_image(canvas_sz, offset_x, offset_y)

    M = cv2.getPerspectiveTransform(warp_src, warp_dst)
    Minv = cv2.getPerspectiveTransform(warp_dst, warp_src)

    return M, Minv
