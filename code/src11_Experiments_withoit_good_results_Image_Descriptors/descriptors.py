#!/usr/bin/python
import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

import skimage.io as skio
from skimage import feature
from skimage.color import rgb2gray

def buildLbp(image, radius, numPoints, mask = None, eps=1e-7):
  """
    
  """
  gray = image
  if (image.shape[2] == 3):
    gray = rgb2gray(image)

  lbp = feature.local_binary_pattern(gray, numPoints,
                                     radius, method="uniform")

  if (mask is not None):
    lbp = lbp[mask]

  # compute the Local Binary Pattern representation
  # of the image, and then use the LBP representation
  # to build the histogram of patterns
  (hist, _) = np.histogram(lbp.ravel(),
                           bins=np.arange(0, numPoints + 3),
                           range=(0, numPoints + 2))

  # normalize the histogram
  hist = hist.astype("float")
  hist /= (hist.sum() + eps)

  # return the histogram of Local Binary Patterns
  return hist

def buildRGBHist(image, bins = 32, mask = None):
  
  hist_r = cv2.calcHist([image[:,:,0]], [0], mask, [bins], [0, 256]).reshape(-1)
  hist_g = cv2.calcHist([image[:,:,1]], [0], mask, [bins], [0, 256]).reshape(-1)
  hist_b = cv2.calcHist([image[:,:,2]], [0], mask, [bins], [0, 256]).reshape(-1)

  return np.stack([hist_r, hist_g, hist_b]).reshape(-1)

def buildKMeansRGBSpaceHist(image, kmeans_model, mask = None):
  rgb_pts = image[mask].reshape(-1, 3)

  prediction = kmeans_model.predict(rgb_pts)

  return np.bincount(prediction, minlength = kmeans_model.n_clusters)

