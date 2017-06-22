#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import pandas as pd
import os
import skimage.morphology as skmorph
import skimage.io as skio
import skimage.transform as sktf
# from skimage.measure import label, regionprops
import skimage.measure as skmes
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from scipy.ndimage.morphology import binary_fill_holes

def get_max_blob_mask(pmsk):
    tlbl = skmes.label(pmsk)
    tprops = skmes.regionprops(tlbl)
    arr_areas = np.array([xx.area for xx in tprops])
    idxMax = np.argmax(arr_areas)
    lblMax = tprops[idxMax].label
    retMsk = (tlbl==lblMax)
    (P0, R) = cv2.minEnclosingCircle(np.array(np.where(retMsk)).transpose())
    return (retMsk, R)

if __name__ == '__main__':
    # fidx = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/02_test-x512-bordered/idx.txt'
    fidx = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/01_train_add-x512-original-bordered/idx.txt'
    wdir = os.path.dirname(fidx)
    prefMskChn = '-msk-CHANNEL.png'
    prefMskCrv = '-msk-cervix.png'
    prefImgMsk = '-automasked.png'
    #
    isDebug = False
    lstPaths = pd.read_csv(fidx)['path'].as_matrix()
    lstPaths = [os.path.join(wdir, xx) for xx in lstPaths]
    numPaths = len(lstPaths)
    errorCounter = 0
    for ipath, fimg in enumerate(lstPaths):
        # if ipath>15:
        #     print ('-')
        fmskChn = '{0}{1}'.format(fimg, prefMskChn)
        fmskCrv = '{0}{1}'.format(fimg, prefMskCrv)
        img = skio.imread(fimg)
        try:
            # (1) load images
            mskChn = skio.imread(fmskChn)[:,:,3].astype(np.float)/255.
            mskCrv = skio.imread(fmskCrv)[:,:,3].astype(np.float)/255.
            # (2) preprocess Cervix Mask
            _, R_crv =  get_max_blob_mask(mskCrv>0.5)
            msizCRV = math.ceil(R_crv * 0.04)
            if msizCRV<3:
                msizCRV = 3
            mskCrv_Blob2 = skmorph.closing(mskCrv>0.5, skmorph.disk(msizCRV))
            mskCrv_Blob3, _ =  get_max_blob_mask(mskCrv_Blob2>0)
            mskCrv_Blob4 = binary_fill_holes(mskCrv_Blob3)
            # (3) preprocess Channel mask
            mskChn_Blob1 = mskChn.copy()
            mskChn_Blob1[~mskCrv_Blob4] = 0
            # (3.1) check zero-channel-mask
            if np.sum(mskChn_Blob1 > 0.5) < 1:
                mskChn_Blob1 = skmorph.skeletonize(mskCrv_Blob4)
                mskChn_Blob1 = skmorph.closing(mskChn_Blob1, skmorph.disk(5))
                R_chn = 5
            else:
                _, R_chn = get_max_blob_mask(mskChn_Blob1>0.5)
            msizChn = math.ceil(R_chn * 0.1)
            if msizChn<3:
                msizChn = 3
            mskChn_Blob2 = skmorph.closing(mskChn_Blob1>0.5, skmorph.disk(msizChn))
            mskChn_Blob2[~mskCrv_Blob4] = 0
            # (3.1) check zero-channel-mask
            if np.sum(mskChn_Blob2 > 0) < 1:
                mskChn_Blob2 = skmorph.skeletonize(mskCrv_Blob4)
                mskChn_Blob2 = skmorph.closing(mskChn_Blob2, skmorph.disk(5))
                #
            mskChn_Blob3, _ = get_max_blob_mask(mskChn_Blob2 > 0)
            mskChn_Blob4 = binary_fill_holes(mskChn_Blob3)
            # (4) Composing output mask
            mskShape = img.shape[:2]
            mskOut = 64 * np.ones(mskShape, dtype=np.uint8)
            mskOut[mskCrv_Blob4] = 255
            mskOut[mskChn_Blob4] = 128
        except Exception as err:
            print ('\t!!! ERROR !!! [{0}]'.format(err))
            errorCounter += 1
            mskOut = np.zeros(img.shape[:2], dtype=np.uint8)
        if img.shape[-1]<4:
            retImg = np.dstack( (img, mskOut) ).astype(np.uint8)
        else:
            retImg = img.copy()
            retImg[:,:,3] = mskOut
        # (5) save results
        foutImg = '{0}{1}'.format(fimg, prefImgMsk)
        if isDebug:
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.subplot(2, 2, 2)
            plt.imshow(mskCrv)
            plt.subplot(2, 2, 3)
            plt.imshow(mskChn)
            plt.subplot(2, 2, 4)
            plt.imshow(mskOut)
            plt.show()
        else:
            print ('[{0}/{1}] :: save masked image -> [{2}]'.format(ipath, numPaths, foutImg))
            skio.imsave(foutImg, retImg)
    print ('\t Total Erros: {0} / {1}'.format(errorCounter, numPaths))
