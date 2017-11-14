# cs231a-project
<img src=https://github.com/rmsare/cs231a-project/raw/master/svm_results.gif height="240px" width="320px">

Code for simple segmentation/classification of UAV images and photogrammetric point clouds. Main functionality:

- Label training or validation images
- Predict pixel labels by SVM or *k*-means
- Project pixel labels onto a point cloud aligned with survey images
- Estimate ground plane orientation of the scene using a subset of classified points

Applied to survey images over an active volcano to identify ground, vegetation and shadow points for CS231A project (Stanford Spring 2017).

Requires skimage, sklearn, numpy, matplotlib, and (optionally) commerical PhotoScan API for photogrammetric point cloud classification.

Contact me ([robertmsare@gmail.com](mailto:robertmsareNO@SPAMgmail.com)) with any questions, bugs, or suggestions. 

## Changelog
Date | Description
---- | -----------
10 June 2017 | Improve commenting, README for submission
9 June 2017  | Functionality complete for final submission
9 June 2017  | Add RANSAC estimation from private repo
8 June 2017  | Update from private repository

## TODO
- More training data
- Conditional random field or DL
- Extend to survey areas with more dense vegetation
- Apply to survey data from August 2017
