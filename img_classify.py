import numpy as np
import matplotlib.pyplot as plt

import skimage.color
from skimage.segmentation import mark_boundaries

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

import os, fnmatch

from img_utils import parse_filename, invert_binary_image, load_image_segments


def classify_all_in_directory(train_dir, test_dir):
    print("Preparing training data...")
    n_samples = 1000
    train_data, train_labels = process_training_images(train_dir, n_samples)
    
    print("Training classifier...")
    classifier = ImageSVC()
    classifier.fit(train_data, train_labels)
    
    files = os.listdir(test_dir)

    for f in files:
        name = parse_filename(f)
        image, segments = load_image_segments(name)
        
        print("Predicting labels for " + name + ".jpg")
        pred_labels = classifier.classify_image(image) 
        
        print("Saving predictions for " + name + ".jpg")
        plot_class_image(image, segments, pred_labels)
        save_prediction(name, pred_labels)
        #save_binary_mask(name, pred_labels)

def evaluate_prediction(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred)
    return C
        
def load_image_data(name):
    fname = 'images/' + name + '_image.npy'
    image = np.load(fname)
    fname = 'labelled/' + name + '_labels.npy'
    labels = np.load(fname)
    fname = 'predict/' + name + '_pred.npy'
    if os.path.exists(fname):
        pred = np.load(fname)
    else:
        pred = None

    return image, labels, pred

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

def process_training_images(train_dir, n_samples):
    n_feat = 3
    train_data = np.empty((0, n_feat))
    train_labels = np.empty(0)
    files = os.listdir(train_dir)

    for f in files:
        name = parse_filename(f)
        image, labels = load_image_labels(name)
        wid, ht, depth = image.shape
        train_data = np.append(train_data, 
                skimage.color.rgb2lab(image).reshape((wid*ht, depth)), axis=0)
        train_labels = np.append(train_labels, 
                labels.reshape(wid*ht, 1).ravel())

    train_data, train_labels = shuffle(train_data, train_labels, 
                                       random_state=0, n_samples=n_samples)
    return train_data, train_labels

def save_binary_mask(name, pred_labels):
    mask = pred_labels
    mask[mask != 1] = 0
    mask = invert_binary_image(mask) # TODO: don't force binary classification 
    elt = np.ones((5, 5))
    mask = skimage.morphology.opening(mask, selem=elt)
    mask = invert_binary_image(mask)
    skimage.io.imsave('masks/' + name + '_mask.png', mask)

def save_prediction(name, pred_labels):
    np.save('predict/' + name + '_pred', pred_labels)

class ImageSVC(SVC):
    
    def classify_image(self, image):
        wid, ht, depth = image.shape
        data = skimage.color.rgb2lab(image).reshape((wid*ht, depth))
        # TODO: compute other features
        pred_labels = self.predict(data)

        return pred_labels.reshape((wid, ht))

