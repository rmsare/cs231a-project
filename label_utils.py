"""
Utilities for segmenting and labelling images
"""

import numpy as np
import matplotlib.pyplot as plt

import skimage.color
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread
from skimage.util import img_as_float

import os
import fnmatch
from copy import copy 
from img_utils import *

def assign_label(x, y, label, segments):
    value = segments[y, x]
    labelled_image[segments == value] = label

# TODO: define callback function without global variables
def label_point(event):
    if event.key == 'q':
        # default: all unlabelled points saved as ground
        labelled_image[np.isnan(labelled_image)] = 1
        save_results(base, image, segments, labelled_image)
        return

    # TODO: more labels: shadow
    labels = { 'g' : 1, # ground
               'v' : 2, # vegetation
               'n' : 0, # null
               'u' : np.nan 
             }
    code = labels

    print("Segment: {} | Label: {}".format(segments[event.ydata, event.xdata], event.key))
    
    assign_label(event.xdata, event.ydata, code[event.key], segments)

    im.set_data(labelled_image)
    fig.canvas.draw()

def label_image(fname):
    ax.imshow(labelled_image, alpha=0.5, vmin=1, vmax=2, cmap=palette)
    plt.show(block=False)

    print("Label points: [g]round, [v]egetation, [n]ull, or [u]ndo. [q] to quit and save.")
    print("Unlabelled points will default to [g]round")

    fig.canvas.mpl_connect('key_press_event', label_point)

def load_data(fname):
    name = parse_filename(fname)
    seg_matches = np.array([fnmatch.fnmatch(file, name + '*_seg.npy') for file in 
                            os.listdir('images/')])
    lab_matches = np.array([fnmatch.fnmatch(file, name + '*_labels.npy') for file in 
                            os.listdir('labelled/')])
    if seg_matches.any():
        image, segments = load_image_segments(name)
    else:
        image = img_as_float(skimage.io.imread(fname)[0])
        factor = 100**2
        n_seg = int(np.prod(image.shape)/factor)
        segments = slic(image, n_segments=n_seg, sigma=5)
    if lab_matches.any():
        fname = 'labelled/' + name + '_labels.npy'
        labelled_image = np.load(fname)
    else:
        labelled_image = np.full(image[:,:,0].shape, np.nan)

    return image, segments, labelled_image

def plot_labelled_image(image, labelled_image):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    base_im = ax.imshow(image)
    palette = copy(plt.cm.Set1)
    palette.set_bad(alpha=0.0)
    im = ax.imshow(labelled_image, alpha=0.75, vmin=1, vmax=2, cmap=palette)
    plt.show(block=False)

def save_results(name, labelled_image):
    np.save('labelled/' + name + '_labels', labelled_image)

if __name__ == "__main__":
    fname = 'data/DJI_0822.JPG'
    image, segments, labelled_image = load_data(fname) 
    
    # TODO: do this wthout global figure and label variables
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    base_im = ax.imshow(mark_boundaries(image, segments, color=(1,0,0), mode='thick'))
    palette = copy(plt.cm.Set1)
    palette.set_bad(alpha=0.0)
    im = ax.imshow(labelled_image, alpha=0.5, vmin=1, vmax=2, cmap=palette)
    plt.show(block=False)

    label_image(fname)
