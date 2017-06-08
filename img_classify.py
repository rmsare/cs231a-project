"""
Classify pixels in images
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


def cluster_image_kmeans(image):
    pass

def classify_directory_svm(classifier, test_dir):
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
        ht, wid, dep = image.shape

        print("Predicting labels for " + f.strip('.JPG') + ".jpg")
        features = skimage.color.rgb2lab(image).reshape((ht*wid, dep))
        features /= features.max(axis=0)
        pred_labels = classifier.predict(features)
 
        
        print("Saving predictions for " + f.strip('.JPG') + ".jpg")
        plt.figure()
        plt.imshow(image)
        plt.imshow(pred_labels.reshape((ht, wid)), alpha=0.5, vmin=0, vmax=2)
        plt.show(block=False)
        plt.savefig('/home/rmsare/results/' + f.strip('.JPG') + '_svm2_pred.png')
        plt.close()
        np.save('/home/rmsare/results/' + f.strip('.JPG') + 'svm2.npy', pred_labels.reshape((ht,wid)))

def load_ground_truth(filename):
    truth = np.load(filename)

    # Change labels for nice default colorscale when plotted
    truth = truth - 1
    truth[truth == -1] = 0
    truth[truth == 0] = 5
    truth[truth == 2] = 0
    truth[truth == 5] = 2

    return truth

def load_image_labels(name):
    fname = 'images/' + name + '_image.npy'
    image = np.load(fname)
    fname = 'labelled/' + name + '_labels.npy'
    labels = np.load(fname)

    return image, labels

def plot_class_image(image, segments, labels):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mark_boundaries(image, segments, color=(1,0,0), mode='thick'))
    plt.title('segmented image')

    plt.subplot(1,2,2)
    plt.imshow(image)
    plt.imshow(labels, alpha=0.75, vmin=1, vmax=2, cmap=plt.cm.Set1)
    cb = plt.colorbar(orientation='horizontal', shrink=0.5)
    plt.title('predicted class labels')
    plt.show(block=False)

def load_training_images(train_dir, n_samples):
    n_feat = 3
    train_data = np.empty((0, n_feat))
    train_labels = np.empty(0)
    files = os.listdir(train_dir)

    for f in files:
        name = parse_filename(f)
        image, labels = load_image_labels(name)
        ht, wid, depth = image.shape
        # TODO: add feature here
        train_data = np.append(train_data, 
                compute_color_features(image), axis=0)
        train_labels = np.append(train_labels, 
                labels.reshape(wid*ht, 1).ravel())

    train_data, train_labels = shuffle(train_data, train_labels, 
                                       random_state=0, n_samples=n_samples)
    return train_data, train_labels

def save_prediction(name, pred_labels):
    np.save('predict/' + name + '_pred', pred_labels)


class Image(object):

    def __init__(self, filename):
        self.imdata = skimage.io.imread(filename)
        self.height, self.width, _ = self.imdata.shape
        self.colorxy = self.compute_colorxy_features()

    def compute_colorxy_features(self):
        colors = skimage.color.rgb2lab(self.imdata)
        X, Y = np.meshgrid(np.arange(self.height), np.arange(self.width))
        xy = np.hstack([X.reshape((self.height*self.width, 1)), Y.reshape((self.height*self.width, 1))])

        colorxy_data = np.hstack([xy, colors])
        colorxy_data /= colorxy_data.max(axis=0)

        return colorxy_data
    
    def compute_glcm_features(self):
        pass

    def compute_gabor_features(self):
        pass

class ImageSVC(SVC):

    def fit_image(self, image):
        self.fit(image.colorxy)

    def classify_image(self, image):
        ht, wid, depth = image.shape
        # TODO: compute other features
        pred_labels = self.predict(image.colorxy)

        return pred_labels.reshape((wid, ht))

