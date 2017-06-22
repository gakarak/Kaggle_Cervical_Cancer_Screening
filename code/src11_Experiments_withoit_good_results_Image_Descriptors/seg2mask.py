#!usr/bin/python
import utilities as utils

import os
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt

def seg2mask(cervix_segmentation_mask, channel_segmentation_mask, 
             min_thresh_cervix = 240, min_thresh_channel = 240):
  assert (cervix_segmentation_mask.shape == channel_segmentation_mask.shape), 'different input mask shapes'

  mask = np.empty_like(cervix_segmentation_mask)
  mask.fill(utils.BACKGROUND_ALPHA)

  cervix_mask = cervix_segmentation_mask > min_thresh_cervix
  channel_mask = channel_segmentation_mask > min_thresh_channel

  mask[cervix_mask] = utils.CERVIX_ALPHA
  mask[np.logical_and(channel_mask, cervix_mask)] = utils.CHANNEL_ALPHA

  return mask

if __name__ == '__main__':
  cervix_segment_idx = '../../dataset/test-x512-bordered/idx_cervix-mask.txt'
  channel_segment_idx = '../../dataset/test-x512-bordered/idx_channel-mask.txt'
  folder_to_save = '../../dataset/train-x512-processed-stage2/test'

  cervix_idx_folder = os.path.split(cervix_segment_idx)[0]
  channel_idx_folder = os.path.split(channel_segment_idx)[0]

  i = 0
  with open(cervix_segment_idx) as cervix_idx, open(channel_segment_idx) as channel_idx:
        #next(cervix_idx) #read header
        #next(channel_idx) #read header
        for cervix_filename, channel_filename in zip(cervix_idx, channel_idx):
            print(i)
            cervix_filename = cervix_filename.strip()
            channel_filename = channel_filename.strip()

            cervix_img = np.array(skio.imread(os.path.join(cervix_idx_folder, cervix_filename)))
            channel_img = np.array(skio.imread(os.path.join(channel_idx_folder, channel_filename)))
            
            mask = seg2mask(cervix_img[:,:,3], channel_img[:,:,3], 220, 128)

            cervix_img[:,:,3] = mask

            image_name = os.path.basename(cervix_filename).split('.')[0] + '.jpg'
            save_name = os.path.join(folder_to_save, image_name + '-processed.png')
            skio.imsave(save_name, cervix_img)
            i+=1