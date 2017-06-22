import os
import numpy as np
import pickle

from descriptor_manager import DescriptorManager
import utilities as utils

if __name__ == '__main__':
  wdir = '../../dataset/train-x512-processed-stage2'
  idx = os.path.join(wdir, 'idx_test.txt')
  manager = DescriptorManager(os.path.join(wdir, 'descriptors'))

  print('Reading data...')

  cervixes = utils.readCervixes(idx, "test")

  print('Building descriptors')

  cervix_kmeans_model = None
  cervix_kmeans_name = 'models/kmeans_pixels-20000_clusters-64'
  with open(cervix_kmeans_name, 'rb') as f:
    cervix_kmeans_model = pickle.load(f)

  channel_kmeans_model = None
  channel_kmeans_name = 'models/kmeans_channel_pixels-20000_clusters-64' 
  with open(channel_kmeans_name, 'rb') as f:
    channel_kmeans_model = pickle.load(f) 

  configs = [
    ['RGB-hist', {'bins': 32, 'mask': None}, ''],
    # ['RGB-hist', {'bins': 128, 'mask': None}, ''],
    # ['RGB-hist', {'bins': 256, 'mask': None}, ''],
    ['LBP', {'radius': 12, 'numPoints': 36, 'mask': None}, ''],
    # ['LBP', {'radius': 16, 'numPoints': 48, 'mask': None}, ''],
    # ['LBP', {'radius': 32, 'numPoints': 160, 'mask': None}, ''],
    ['RGB-hist', {'bins': 32, 'mask': None}, 'channel-mask'],
    ['RGB-hist', {'bins': 128, 'mask': None}, 'channel-mask'],
    ['RGB-hist', {'bins': 256, 'mask': None}, 'channel-mask'],
    ['LBP', {'radius': 12, 'numPoints': 36, 'mask': None}, 'channel-mask'],
    ['LBP', {'radius': 16, 'numPoints': 48, 'mask': None}, 'channel-mask'],
    ['LBP', {'radius': 32, 'numPoints': 160, 'mask': None}, 'channel-mask'],
    # ['KMeans', {'kmeans_model': kmeans_model, 'mask': None}, "cervix-20000-512"],
    # ['KMeans', {'kmeans_model': kmeans_model, 'mask': None}, "cervix-20000-256"],
    ['KMeans', {'kmeans_model': cervix_kmeans_model, 'mask': None}, "cervix-20000-64"],
    # ['KMeans', {'kmeans_model': kmeans_model, 'mask': None}, "cervix-20000-16"],
    # ['KMeans', {'kmeans_model': kmeans_model, 'mask': None}, "cervix-50000-64"],
    ['KMeans', {'kmeans_model': channel_kmeans_model, 'mask': None}, "channel-20000-64"],
  ]

  for idx,cervix in enumerate(cervixes):
    print('[%d / %d] %s' % (idx+1, len(cervixes), cervix['code_name']))

    for config in configs:
      params = config[1]

      mask = cervix['cervix_mask'].astype(np.uint8)
      if ('channel' in config[2]):
        mask = cervix['channel_mask'].astype(np.uint8)

      if (config[0] == 'KMeans'):
        mask = np.dstack((mask, mask, mask)).astype(np.bool)

      params['mask'] = mask

      vector = manager.calculateFeatures(cervix['image'], config)
      manager.save(cervix['code_name'], config, vector)