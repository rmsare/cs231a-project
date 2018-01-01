"""
Utility functions for processing images

Author: Robert Sare
E-mail: rmsare@stanford.edu
Date: 8 June 2017
"""

import numpy as np
import matplotlib.pyplot as plt

import skimage.color
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread
from skimage.util import img_as_float

import os
import fnmatch

def load_image_segments(name):
    fname = 'images/' + name + '_image.npy'
    image = np.load(fname)
    fname = 'images/' + name + '_seg.npy'
    segments = np.load(fname)
    
    return image, segments

def parse_filename(fname):
    fname = fname.split('/')
    m = len(fname)
    fname = fname[m-1]
    name = fname.split('.')[0]

    return name

def save_results(name, image, segments):
    np.save('images/' + name + '_image', image)
    np.save('images/' + name + '_seg', segments)

def segment_all_in_directory(dirname, factor, sigma):
    files = os.listdir(dirname)

    for filename in files:
        if fnmatch.fnmatch(filename, '*.JPG'):
            print("Processing " + filename + "...")
            name = parse_filename(filename)
            image = img_as_float(imread(dirname + filename)[0])
            n_seg = int(np.prod(image[:,:,0].shape) / factor)
            segments = slic(image, n_segments=n_seg, sigma=sigma)
            save_results(name, image, segments)
