"""
Classification of pixels in images using color features.

Project uses the following directory structure:

    images/     - contains binary files of numpy arrays corresponding to survey images
    labelled/   - contains labelled ground truth images or training data
    results/    - contains results of classification

Author: Robert Sare
E-mail: rmsare@stanford.edu
Date: 8 June 2017
"""

import numpy as np
import matplotlib.pyplot as plt

import skimage.color, skimage.io
from skimage.segmentation import mark_boundaries

from sklearn.svm import SVC
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

import os, fnmatch

def classify_directory(classifier, test_dir, train_dir='train/'):
    """
    Classify all images in a directory using an arbitrary sklearn classifier.  

    Saves results to results/ directory.
    """

    # XXX: This is here if the classifier needs to be trained from scratch
    #print("Preparing training data...")
    #n_samples = 1000
    #train_data, train_labels = load_training_images(train_dir, n_samples)
    #
    #print("Training classifier...")
    #classifier = ImageSVC()
    #classifier.fit(train_data, train_labels)
    
    files = os.listdir(test_dir)

    for f in files:
        image = skimage.io.imread(f)
        height, width, depth = image.shape

        print("Predicting labels for " + f.strip('.JPG') + ".jpg")
        features = compute_colorxy_features(image) 
        features /= features.max(axis=0)
        pred_labels = classifier.predict(features)
 
        
        print("Saving predictions for " + f.strip('.JPG') + ".jpg")
        plt.figure()
        plt.imshow(image)
        plt.imshow(pred_labels.reshape((height, width)), alpha=0.5, vmin=0, vmax=2)
        plt.show(block=False)
        plt.savefig('results/' + f.strip('.JPG') + '_svm_pred.png')
        plt.close()
        np.save('results/' + f.strip('.JPG') + 'svm.npy', pred_labels.reshape((height,width)))

def compute_colorxy_features(image):
    """
    Extract and normalize color and pixel location features from image data
    """

    height, width, depth = image.shape
    colors = skimage.color.rgb2lab(image.reshape((height*width, depth))
    X, Y = np.meshgrid(np.arange(height), np.arange(width))
    xy = np.hstack([X.reshape((height*width, 1)), Y.reshape((height*width, 1))])

    colorxy = np.hstack([xy, colors])
    colorxy /= colorxy.max(axis=0)
    
    return colorxy 

def load_ground_truth(filename):
    """
    Load ground truth or training image array and redefine labelling for nice
    default colors
    """

    truth = np.load(filename)

    # Change labels for nice default colorscale when plotted
    truth = truth - 1
    truth[truth == -1] = 0
    truth[truth == 0] = 5
    truth[truth == 2] = 0
    truth[truth == 5] = 2

    return truth

def load_image_labels(name):
    """
    Load image and labels from previous labelling session
    """

    fname = 'images/' + name + '_image.npy'
    image = np.load(fname)
    fname = 'labelled/' + name + '_labels.npy'
    labels = np.load(fname)

    return image, labels

def plot_class_image(image, segments, labels):
    """
    Display image with segments and class label overlay
    """

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mark_boundaries(image, segments, color=(1,0,0), mode='thick'))
    plt.title('segmented image')

    plt.subplot(1,2,2)
    plt.imshow(image)
    plt.imshow(labels, alpha=0.75)
    cb = plt.colorbar(orientation='horizontal', shrink=0.5)
    plt.title('predicted class labels')
    plt.show(block=False)

def load_training_images(train_dir, n_samples=1000, n_features=3):
    """
    Load training images from directory and subsample for training or validation 
    """

    train_data = np.empty((0, n_features))
    train_labels = np.empty(0)
    files = os.listdir(train_dir)

    for f in files:
        name = parse_filename(f)
        image, labels = load_image_labels(name)
        ht, wid, depth = image.shape
        train_data = np.append(train_data, 
                compute_color_features(image), axis=0)
        train_labels = np.append(train_labels, 
                labels.reshape(wid*ht, 1).ravel())

    train_data, train_labels = shuffle(train_data, train_labels, 
                                       random_state=0, n_samples=n_samples)
    return train_data, train_labels

def save_prediction(name, pred_labels):
    """
    Save predicted class labels 
    """
    np.save('results/' + name + '_pred', pred_labels)

