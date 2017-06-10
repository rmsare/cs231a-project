"""
Utilities for labelling images

Usage: 
1. Change name variable to specify image to label (e.g., "DJI_0825" for DJI_0825.JPG)
2. Hover cursor over segment to label
3. Press key corresponding to label ([g]round, [v]egetation, [s]hadow, ...)
4. Change labels or [u]ndo as necessary
5. Quit and save with [q]

By default, all unlabelled pixels are assigned to ground class. Results are saved to 'labelled/' directory.

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

from copy import copy 

def assign_label(x, y, label, segments):
    """
    Assign label to segment 
    """
    value = segments[y, x]
    labelled_image[segments == value] = label

def label_point(event):
    """
    Label segment under cursor at key press event.
    Supported labels:
        [g]round
        [v]egetation
        [s]hadow
        [n]ull

    Quits on "q" keypress.
    """

    if event.key == 'q':
        # Default: all unlabelled points saved as ground
        labelled_image[np.isnan(labelled_image)] = 1
        save_results(base, image, segments, labelled_image)
        return

    labels = { 'g' : 1, # ground
               'v' : 2, # vegetation
               's' : 3, # shadow
               'n' : 0, # null
               'u' : np.nan 
             }

    print("Keypress registered")
    print("Segment: {} | Label: {}".format(segments[int(event.ydata), int(event.xdata)], event.key))
    
    assign_label(int(event.xdata), int(event.ydata), labels[event.key], segments)

    im.set_data(labelled_image)
    plt.draw()

def label_image():
    """
    Label image until callback function exits.
    """

    print("Label points: [g]round, [v]egetation, [s]hadow, [n]ull, or [u]ndo. [q] to quit and save.")
    print("Unlabelled points will default to [g]round")

    # TODO: parameterize this callback to avoid global variables?
    fig.canvas.mpl_connect('key_press_event', label_point)

def load_image_segments(name):
    """
    Load image and segmentation corresponding to base name "name"
    """

    fname = 'images/' + name + '_image.npy'
    image = np.load(fname)
    fname = 'images/' + name + '_seg.npy'
    segments = np.load(fname)
    
    return image, segments

def save_results(name, labelled_image):
    np.save('labelled/' + name + '_labels', labelled_image)

if __name__ == "__main__":
    # Load image to label
    name = 'DJI_0820'
    image, segments = load_image_segments(name)
    labelled_image = np.full(image[:,:,0].shape, np.nan)

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    base_im = ax.imshow(mark_boundaries(image, segments, color=(1,0,0), mode='thick'))
    palette = copy(plt.cm.viridis)
    palette.set_bad(alpha=0.0)
    im = ax.imshow(labelled_image, alpha=0.5, vmin=1, vmax=2, cmap=palette)
    plt.show(block=False)

    # Label image, updating plot until user quits
    label_image()
