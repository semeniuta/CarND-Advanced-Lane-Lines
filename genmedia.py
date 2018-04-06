'''
Generate the output images and videos, including rendering of the pipeline
'''

import os
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
import cv2
from glob import glob
from moviepy.editor import VideoFileClip
from networkx.drawing.nx_agraph import to_agraph

import lanelines
import calib
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
    cm,
    dc,
    memory_size=25,
    diff_threshold=int(1e4)
):

    '''
    Create a function object for processing each frame of
    a  video file. The returned function accepts an image,
    performs lane detection on it, and returns an image with rendered
    lane region and text on curvature and offset from center measurements.
    '''

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

    def process(frame):

        im = cv2.undistort(frame, cm, dc)

        coefs = coefs_smoother(im)

        rendered_im = lanelines.render_lane(
            im, runner['warped'], coefs['p_coefs_left'], coefs['p_coefs_right'], M_inv
        )

        curvature = curv_smoother(coefs)

        m_offset, _ = lanelines.lane_offset_from_center(
            runner['warped'],
            coefs['p_coefs_left'],
            coefs['p_coefs_right'],
            mx
        )

        offset_direction = 'left' if m_offset > 0 else 'right'

        curv_text = 'Curvature: {:.2f} m'.format(curvature)
        offset_text = 'Offset from center: {:.2f} m (to the {})'.format(np.abs(m_offset), offset_direction)

        lanelines.put_text_on_top(rendered_im, curv_text, fontscale=1.2)
        lanelines.put_text_on_top(rendered_im, offset_text, fontscale=1.2, pos=(10, 120))

        return rendered_im

    return process


def process_images(im_filenames, cg, cg_params, M, M_inv, cm, dc):
    '''
    Process images with the lane detection pipeline
    and return the corresponding images with rendered
    lane region and text on curvature and offset from center measurements
    '''

    runner = CompGraphRunner(cg, frozen_tokens=cg_params)

    def process(im_orig):

        im = cv2.undistort(im_orig, cm, dc)

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
    '''
    Visualize lane detection pipeline using Graphviz
    and save the resulting image on disk
    '''

    runner = CompGraphRunner(cg, frozen_tokens=params)

    ag = to_agraph(runner.token_manager.to_networkx())
    ag.layout('dot')
    ag.draw(fname_dst)


def visualize_chessboard_warp(fname_src, fname_dst, cm, dc, nx=9, ny=6):

    im = mpimg.imread(fname_src)

    found_cbc, cbc, undist_im = calib.undistort_cb_and_find_corners(im, cm, dc, nx, ny)

    im_sz = lanelines.get_im_wh(undist_im)

    canvas = np.copy(undist_im)
    cv2.drawChessboardCorners(canvas, (nx, ny), cbc, found_cbc)

    corners = cbc.reshape(-1, 2)

    src_points = lanelines.get_rectangle_corners_from_cbc(corners, nx, ny)

    offset_x = im_sz[0] / (nx + 1)
    offset_y = im_sz[1] / (ny + 1)

    dst_points = lanelines.get_rectangle_corners_in_image(im_sz, offset_x, offset_y)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(canvas, M, im_sz, flags=cv2.INTER_LINEAR)

    mpimg.imsave(fname_dst, warped)


def visualize_road_undistort(fname_src, fname_dst, cm, dc):

    im = mpimg.imread(fname_src)
    im_undist = cv2.undistort(im, cm, dc)

    mpimg.imsave(fname_dst, im_undist)


def visualize_pipeline_intermediate_results(fname_dst, M, M_inv, cm, dc):

    runner = CompGraphRunner(COMP_GRAPH, frozen_tokens=DEFAULT_PARAMS)

    test_images = [mpimg.imread(f) for f in glob('test_images/*.jpg')]
    test_images_undist = [cv2.undistort(im, cm, dc) for im in test_images]

    plt.figure(figsize=(20, 5))
    for i, im in enumerate(test_images_undist):

        runner.run(image=im, M=M)

        plt.subplot(1, 8, i+1)
        plt.imshow( runner['all_thresholds'])
        _ = plt.axis('off')

        poly_y, poly_x_left, poly_x_right = lanelines.get_lane_polynomials_points(
            runner['warped'],
            runner['p_coefs_left'],
            runner['p_coefs_right']
        )

        plt.plot(poly_x_left, poly_y, color='c')
        plt.plot(poly_x_right, poly_y, color='c')

    plt.tight_layout()
    plt.savefig(fname_dst)


if __name__ == '__main__':

    ''' INITIALIZATION '''

    cm = np.load('serialize/camera_matrix.npy')
    dc = np.load('serialize/dist_coefs.npy')

    im_dir_src = 'test_images'
    im_files_src = get_full_paths_to_files(im_dir_src, os.listdir(im_dir_src))

    video_files = ('project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4')
    video_files_src = get_full_paths_to_files('.', video_files)

    dir_dst = 'output_images'
    create_dir(dir_dst)

    im_files_dst = get_full_paths_to_files(dir_dst, os.listdir(im_dir_src))
    video_files_dst = get_full_paths_to_files(dir_dst, video_files)


    ''' MEDIA GENERATION '''

    M, Minv = prepare_perspective_transforms_custom()
    process = create_processing_func(COMP_GRAPH, DEFAULT_PARAMS, M, Minv, cm, dc)

    visualize_pipeline(os.path.join(dir_dst, 'pipeline.png'))

    visualize_chessboard_warp(
        'camera_cal/calibration10.jpg',
        os.path.join(dir_dst, 'cb_warped.jpg'),
        cm,
        dc
    )

    visualize_road_undistort(
        'test_images/test4.jpg',
        os.path.join(dir_dst, 'road_undist.jpg'),
        cm,
        dc
    )

    visualize_pipeline_intermediate_results(
        os.path.join(dir_dst, 'intermediate.jpg'),
        M,
        Minv,
        cm,
        dc
    )

    images_dst = process_images(im_files_src, COMP_GRAPH, DEFAULT_PARAMS, M, Minv, cm, dc)
    save_images(images_dst, im_files_dst)

    process_and_save_video(video_files_src[0], video_files_dst[0], process)
