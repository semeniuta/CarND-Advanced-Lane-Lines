import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from compgraph import CompGraph, CompGraphRunner

import lanelines

class LaneFinder:

    def __init__(self, cg, params):
        self.cg = cg
        self.runner = CompGraphRunner(cg, frozen_tokens=params)

    def process(self, im):
        self.runner.run(image=im)
        return self['avg_line_left'], self['avg_line_right']

    def __getitem__(self, token_name):
        return self.runner.token_value(token_name)


def create_objects(cg, frozen, line_color=[255, 0, 0], line_thickness=5, alpha=0.8, beta=1.):

    finder = LaneFinder(cg, frozen)

    def find_and_draw_lanes(im):

        left, right = finder.process(im)

        canvas = np.zeros_like(im)
        lanelines.draw_line(canvas, left, color=line_color, thickness=line_thickness)
        lanelines.draw_line(canvas, right, color=line_color, thickness=line_thickness)
        res_im = lanelines.weighted_img(im, canvas, alpha, beta, 0)

        return res_im

    return finder, find_and_draw_lanes
