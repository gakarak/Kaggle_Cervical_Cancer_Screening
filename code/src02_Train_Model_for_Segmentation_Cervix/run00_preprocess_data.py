#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import shutil
import os
import pandas as pd
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import cv2

##########################################################
def addBoundaries(pimg, newShape=(512, 512), isDebug = False):
    tsiz = pimg.shape[:2]
    matShift = np.zeros((2, 3))
    matShift[0, 0] = 1.0
    matShift[0, 1] = 0.0
    matShift[1, 0] = 0.0
    matShift[1, 1] = 1.0
    matShift[0, 2] = +(float(newShape[1] - tsiz[1])) / 2.0
    matShift[1, 2] = +(float(newShape[0] - tsiz[0])) / 2.0
    ret = cv2.warpAffine(pimg, matShift, newShape, None, cv2.INTER_NEAREST)
    if isDebug:
        plt.subplot(2, 2, 1)
        plt.imshow(pimg[:, :, :3])
        plt.title('Inp image = {0}'.format(pimg.shape))
        plt.subplot(2, 2, 2)
        plt.imshow(pimg[:, :, 3])
        plt.title('Inp mask = {0}'.format(pimg.shape[:2]))
        plt.subplot(2, 2, 3)
        plt.imshow(ret[:, :, :3])
        plt.title('Out image = {0}'.format(ret.shape))
        plt.subplot(2, 2, 4)
        plt.imshow(ret[:, :, 3])
        plt.title('Out mask = {0}'.format(ret.shape[:2]))
        plt.show()
    return ret

def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

##########################################################
if __name__ == '__main__':
    # fidx = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/00-data-original-proportions/idx.txt'
    # fidx = '/mnt/data6T/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/02_test-x512/idx.txt'
    fidx = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/00_train_add-x512-original-proportions/idx.txt'
    wdir = os.path.dirname(fidx)
    # odir = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/01-data-512x512'
    # odir = '/mnt/data6T/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/02_test-x512-bordered'
    odir = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/01_train_add-x512-original-bordered'
    make_dirs(odir)
    outShape = (512, 512)
    #
    pathImgs = pd.read_csv(fidx)['path'].as_matrix()
    pathImgsAbs = np.array([os.path.join(wdir, xx) for xx in pathImgs])
    #
    numImgs = len(pathImgs)
    for ipath, path in enumerate(pathImgsAbs):
        relPath = pathImgs[ipath]
        timg = skio.imread(path)
        timgMod = addBoundaries(timg, newShape=outShape, isDebug=False)
        todir = os.path.join(odir, os.path.dirname(relPath))
        foutImg = os.path.join(todir, os.path.basename(relPath))
        make_dirs(todir)
        skio.imsave(foutImg, timgMod)
        print ('[{0}/{1}] -> ({2})'.format(ipath, numImgs, foutImg))
    fidxOut = os.path.join(odir, os.path.basename(fidx))
    print ('::copy Index file: [{0}]'.format(fidxOut))
    shutil.copy(fidx, fidxOut)
