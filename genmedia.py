'''
Generate the output images and videos, including rendering of the pipeline
'''

import os
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from networkx.drawing.nx_agraph import to_agraph

import lanelines
import newlanespipeline
from compgraph import CompGraph, CompGraphRunner
from roadplane import prepare_perspective_transforms_custom
from smooth import Smoother, GenericSmoother, GenericSmootherWithMemory, Memory, compute_diffs

COMP_GRAPH = newlanespipeline.computational_graph
DEFAULT_PARAMS = newlanespipeline.parameters


def create_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

def get_full_paths_to_files(files_dir, filenames):

    return [os.path.join(files_dir, f) for f in filenames]


def create_processing_func(
    cg,
    cg_params,
    M,
    M_inv,
    memory_size=10,
    diff_threshold=int(1e4)
):

    runner = CompGraphRunner(cg, frozen_tokens=cg_params)

    tokens = ('p_coefs_left', 'p_coefs_right')
    thresholds = {
        'p_coefs_left': np.array([0.8e-4, 0.1, 80.]),
        'p_coefs_right': np.array([0.8e-4, 0.1, 80.])
    }

    mx, my = lanelines.pixel_to_meter_ratios_custom()

    coefs_smoother = Smoother(runner, M, tokens, thresholds)

    def curv(coefs):
        return lanelines.lane_curvature(coefs['p_coefs_left'], coefs['p_coefs_right'], mx, my, runner['canvas_size'])

    curv_smoother = GenericSmootherWithMemory(curv, diff_threshold, memory_size)

    def process(im):

        coefs = coefs_smoother(im)

        rendered_im = lanelines.render_lane(
            im, runner['warped'], coefs['p_coefs_left'], coefs['p_coefs_right'], M_inv
        )

        curvature = curv_smoother(coefs)

        im_text = 'Curvature: {:.2f} m'.format(curvature)
        lanelines.put_text_on_top(rendered_im, im_text)

        return rendered_im

    return process


def process_images(im_filenames, cg, cg_params, M, M_inv):

    runner = CompGraphRunner(cg, frozen_tokens=cg_params)

    def process(im):

        runner.run(image=im, M=M)

        rendered = lanelines.render_lane(
            im, runner['warped'], runner['p_coefs_left'], runner['p_coefs_right'], M_inv
        )

        return rendered

    images = (mpimg.imread(fname) for fname in im_filenames)

    return (process(im) for im in images)


def save_images(images, destination_filenames):

    for fname, im in zip(destination_filenames, images):
        mpimg.imsave(fname, im)


def process_and_save_video(video_fname_src, video_fname_dst, processing_func):

    video_src = VideoFileClip(video_fname_src)

    video_dst = video_src.fl_image(processing_func)
    video_dst.write_videofile(video_fname_dst, audio=False)


def visualize_pipeline(fname_dst, cg=COMP_GRAPH, params=DEFAULT_PARAMS):

    runner = CompGraphRunner(cg, frozen_tokens=params)

    ag = to_agraph(runner.token_manager.to_networkx())
    ag.layout('dot')
    ag.draw(fname_dst)


if __name__ == '__main__':

    ''' INITIALIZATION '''

    im_dir_src = 'test_images'
    im_files_src = get_full_paths_to_files(im_dir_src, os.listdir(im_dir_src))

    video_files = ('project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4')
    video_files_src = get_full_paths_to_files('.', video_files)

    dir_dst = 'output'
    create_dir(dir_dst)

    im_files_dst = get_full_paths_to_files(dir_dst, os.listdir(im_dir_src))
    video_files_dst = get_full_paths_to_files(dir_dst, video_files)


    ''' MEDIA GENERATION '''

    M, Minv = prepare_perspective_transforms_custom()
    process = create_processing_func(COMP_GRAPH, DEFAULT_PARAMS, M, Minv)

    visualize_pipeline(os.path.join(dir_dst, 'pipeline.png'))

    images_dst = process_images(im_files_src, COMP_GRAPH, DEFAULT_PARAMS, M, Minv)
    save_images(images_dst, im_files_dst)

    process_and_save_video(video_files_src[0], video_files_dst[0], process)
