'''
Generate the output images and videos, including rendering of the pipeline
'''

import os
import matplotlib.image as mpimg
import cv2
from moviepy.editor import VideoFileClip
from networkx.drawing.nx_agraph import to_agraph

import lanespipeline
import lanefinder
from compgraph import CompGraph, CompGraphRunner


COMP_GRAPH = lanespipeline.computational_graph
DEFAULT_PARAMS = lanespipeline.parameters


def create_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

def get_full_paths_to_files(files_dir, filenames):

    return [os.path.join(files_dir, f) for f in filenames]


def process_images(im_filenames, cg, params):

    finder, find_and_draw_lanes = lanefinder.create_objects(cg, params)

    images = (mpimg.imread(fname) for fname in im_filenames)

    return (find_and_draw_lanes(im) for im in images)


def save_images(images, destination_filenames):

    for fname, im in zip(destination_filenames, images):
        mpimg.imsave(fname, im)


def process_and_save_video(video_fname_src, video_fname_dst, cg, params):

    finder, find_and_draw_lanes = lanefinder.create_objects(cg, params)

    video_src = VideoFileClip(video_fname_src)

    video_dst = video_src.fl_image(find_and_draw_lanes)
    video_dst.write_videofile(video_fname_dst, audio=False)


def visualize_pipeline(fname_dst, cg=COMP_GRAPH, params=DEFAULT_PARAMS):

    runner = CompGraphRunner(cg, frozen_tokens=params)

    ag = to_agraph(runner.token_manager.to_networkx())
    ag.layout('dot')
    ag.draw(fname_dst)


if __name__ == '__main__':

    ''' INITIALIZATION '''

    im_dir_src = 'test_images'
    im_dir_dst = 'test_images_output'
    create_dir(im_dir_dst)

    im_files_src = get_full_paths_to_files(im_dir_src, os.listdir(im_dir_src))
    im_files_dst = get_full_paths_to_files(im_dir_dst, os.listdir(im_dir_src))

    video_dir_src = 'test_videos'
    video_dir_dst = 'test_videos_output'
    create_dir(video_dir_dst)

    video_files = ('solidWhiteRight.mp4', 'solidYellowLeft.mp4')
    video_files_src = get_full_paths_to_files(video_dir_src, video_files)
    video_files_dst = get_full_paths_to_files(video_dir_dst, video_files)

    params_1 = DEFAULT_PARAMS.copy()
    params_1['canny_lo'] = 50
    params_1['canny_hi'] = 150

    ''' MEDIA GENERATION '''

    visualize_pipeline('pipeline.png')

    images_dst = process_images(im_files_src, COMP_GRAPH, DEFAULT_PARAMS)
    save_images(images_dst, im_files_dst)

    process_and_save_video(video_files_src[0], video_files_dst[0], COMP_GRAPH, DEFAULT_PARAMS)
    process_and_save_video(video_files_src[1], video_files_dst[1], COMP_GRAPH, params_1)
