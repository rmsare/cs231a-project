"""
Point-cloud processing and ground plane estimation utilities.

Author: Robert Sare
E-mail: rmsare@stanford.edu
Date: 8 June 2017

Some functions equire PhotoScan library and must be run in PhotoScan IPython 
console. See API documentation at:

http://www.agisoft.com/pdf/photoscan_python_api_1_3_2.pdf

Import statements for this library are included within function definitions so
other processing can take place outside PhotoScan. 
"""

from sklearn import linear_model
import numpy as np

def load_class_labels(name):
    return np.load('predicted/' + name + '_pred.npy')

def load_point_cloud(filename, npoints=1000, ncols=9):
    """
    Load point cloud from text format. Points are stored in lines as plaintext:
    
        X Y Z R G B NX NY NZ ...

    where R, G, B are color channels (0, 255) and NX, NY, NZ denote the 
    components of the point normal in a local coordinate system.
    """

    data = np.fromfile(open(filename, 'r'), sep=' ', count=npoints*ncols)
    data = data.reshape((npoints, ncols))
    nrows, ncols = data.shape
    points = data[:, 0:3]
    rgb_values = data[:, 3:6].reshape((nrows, 1, 3))
    features = skimage.color.rgb2lab(rgb_values.astype(np.uint8)).reshape((nrows, 3))
    features /= features.max(axis=0)
    return points, features

def classify_points(classifier, points, features):
    labels = clf.predict(features)
    classified_points = np.hstack([points, labels])
    return classified_points

def estimate_ground_plane(classified_points, ground_label=2, thresh=0.1):
    """
    Estimate ground plane from classified point cloud
    """

    ground_points = classified_points[classified_points[:,3] == ground_label]

    XY = ground_points[:,0:2]
    Z = ground_points[:,2]
    
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),
                                          residual_threshold=thresh)
    ransac.fit(XY, Z)
    
    return ransac.estimator_.coef_, ransac.inlier_mask_, np.logical_not(ransac.inlier_mask_)

def get_modal_point_label(point, cameras):
    """
    Compute distribution of class labels for a 3D points from classified images.

    Returns:
        label_dist - distribution of class labels from corresponding 2D points
    """
    import PhotoScan

    label_dist = []
    for camera in cameras:
        name = camera.label.strip('.JPG')
        class_labels = load_class_labels(name)
        
        image_height = int(camera.meta['File/ImageHeight'])
        image_width = int(camera.meta['File/ImageWidth'])
        
        x, y = camera.project(point)
        x_in_image = x >= 0 and x < image_width
        y_in_image = y >= 0 and y < image_height
        if x_in_image and y_in_image:
            label_dist.append(class_labels[y, x])

    return np.array(label_dist)

def project_features(camera, points, features):
    """
    Project feature values from 3D points onto an image using the camera matrix.

    Returns:
        projected_features - an array of (image_height, image_width) of feature
                             values corresponding to pixels in the image
    """
    import PhotoScan
    image_height = int(camera.meta['File/ImageHeight'])
    image_width = int(camera.meta['File/ImageWidth'])
    projected_features = np.zeros_like((image_height, image_width))

    for i, P in enumerate(points):
        P = PhotoScan.Vector(tuple(P))
        x, y = camera.project(P)
        x_in_image = x >= 0 and x < image_width
        y_in_image = y >= 0 and y < image_height
        if x_in_image and y_in_image:
            projected_features[y, x] = features[i]

    return projected_features.reshape((image_height, image_width))

def read_camera_matrix(filename):
    """
    Read camera matrix from text file exported by PhotoScan
    """
    f = open(filename, 'r')
    s = f.read()
    f.close()

    s = s.split(',')
    s = [x.strip('\Matrix([[') for x in s]
    s = [x.strip(']])') for x in s]
    s = [x.strip('[') for x in s]
    s = [x.strip(']') for x in s]
    s = [x.strip('\n      [') for x in s]
    M = np.array([float(x) for x in s])
    
    return M.reshape((4,4))

if __name__ == "__main__":
    doc = PhotoScan.app.doc
    cam = doc.chunks[0].cameras[0]
    filename = '/home/rmsare/horseshoe_dense.txt'
    points, features = load_point_cloud(filename)
    proj = project_features(cam, points, features)
