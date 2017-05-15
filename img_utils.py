"""
Utility functions for processing images
"""

import numpy as np
import matplotlib.pyplot as plt

import skimage.color
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread
from skimage.util import img_as_float

import os
import fnmatch

def downsample_slic(image, factor, sigma):

    n_pixels = int(np.prod(image[:,:,0].shape)/factor)
    segments = slic(image, n_segments=n_pixels, sigma=sigma)
    
    R = np.zeros((n_pixels, 1))
    G = np.zeros((n_pixels, 1))
    B = np.zeros((n_pixels, 1))
    I = np.zeros((n_pixels, 1))
    J = np.zeros((n_pixels, 1))

    labels = np.unique(segments)
    for i, value in enumerate(labels):
        rows, cols = np.where(segments == value)
        
        R[i] = np.mean(image[rows, cols, 0])
        G[i] = np.mean(image[rows, cols, 1])
        B[i] = np.mean(image[rows, cols, 2])
        I[i] = np.mean(rows) 
        J[i] = np.mean(cols)
    
    m, n = image[:,:,0].shape
    s = (int(m/np.sqrt(factor)), int(n/np.sqrt(factor)))
    coarse_image = np.stack([R.reshape(s), G.reshape(s), B.reshape(s)], axis=-1)
    centroids = np.stack([I.reshape(s), J.reshape(s)], axis=-1)

    return centroids, segments, coarse_image

def invert_binary_image(image):
    ones = image == 1
    zeros = image == 0
    image[ones] = 0
    image[zeros] = 1
    return image

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

    for file in files:
        if fnmatch.fnmatch(file, '*.JPG'):
            print("Processing " + file + "...")
            name = parse_filename(file)
            image = img_as_float(imread(dirname + file)[0])
            n_seg = int(np.prod(image[:,:,0].shape)/factor)
            segments = slic(image, n_segments=n_seg, sigma=sigma)
            save_results(name, image, segments)
