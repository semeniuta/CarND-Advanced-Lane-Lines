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


if __name__ == '__main__':

    im_straight_1 = lanelines.open_image('test_images/straight_lines1.jpg')
    im_straight_2 = lanelines.open_image('test_images/straight_lines2.jpg')

    points = define_flat_plane_on_road(
        (im_straight_1, im_straight_2),
        x_offset=0
    )

    print('Road plane points:')
    print(points)

    np.save('serialize/planepoints.npy', points)
