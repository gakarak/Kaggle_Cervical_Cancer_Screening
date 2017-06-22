#!usr/bin/python
import os
import pickle
import random
import numpy as np
from sklearn.cluster import KMeans

import utilities as utils

if __name__ == '__main__':
  wdir = '../../dataset/train-x512-processed-stage2'
  idx = os.path.join(wdir, 'idx_all.txt')

  print('Reading data...')
  cervixes = utils.readCervixes(idx)

  REQUIRED_PIXELS = 20000
  print('Taking %d pixels...' % REQUIRED_PIXELS)
  PIXELS_PER_IMG = (REQUIRED_PIXELS // len(cervixes)) + 1

  random.seed(42)
  training_pixels = []
  for c in cervixes:
    mask = c['cervix_mask']
    mask = np.dstack((mask, mask, mask))
    pixels = c['image'][mask].reshape(-1, 3)
    for _ in range(PIXELS_PER_IMG):
      rnd = random.randint(0, pixels.shape[0] - 1)
      training_pixels.append(pixels[rnd])


  CLUSTER_COUNT = 64
  print('Training %d clusters...' % CLUSTER_COUNT)
  kmeans = KMeans(n_clusters=CLUSTER_COUNT, random_state=42).fit(training_pixels)

  outfile = 'models/kmeans_pixels-%d_clusters-%d' % (REQUIRED_PIXELS, CLUSTER_COUNT)
  with open(outfile, 'wb') as pickle_file:
    pickle.dump(kmeans, pickle_file)
  print('Success')