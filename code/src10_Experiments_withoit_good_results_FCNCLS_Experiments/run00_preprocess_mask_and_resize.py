#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import shutil
import os
import pandas as pd
import numpy as np
import skimage.io as skio
import skimage.transform as sktf
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

def resizeToMaxSize(pimg, poutSize, porder=2):
    pshape = pimg.shape[:2]
    nrow, ncol = pshape
    if nrow>=ncol:
        newShape = (poutSize, int(float(poutSize) * ncol / nrow))
    else:
        newShape = ((int(float(poutSize) * nrow / ncol)), poutSize)
    ret = sktf.resize(pimg, newShape, order=porder, preserve_range=True).astype(pimg.dtype)
    return ret

def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

##########################################################
if __name__ == '__main__':
    isDebug = False
    # isAddBorder = True
    isAddBorder = False
    outSize = 512 #+ 512
    #
    fidxInp = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-original/idx.txt'
    dirInp = os.path.dirname(fidxInp)
    dirMasked = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/00-data-original-proportions'
    dirOut = '{0}-{1}x{1}'.format(dirInp, outSize)
    if isAddBorder:
        dirOut = '{0}-bordered'.format(dirOut)
    make_dirs(dirOut)
    outShape = (outSize, outSize)
    # (1) input image paths
    pathInpImgs = pd.read_csv(fidxInp)['path'].as_matrix()
    pathInpImgsAbs = np.array([os.path.join(dirInp, xx) for xx in pathInpImgs])
    # (2) input masks paths
    pathInpMasks = np.array([os.path.join(dirMasked, '{0}-msk.png-channel.png'.format(xx)) for xx in pathInpImgs])
    #
    numImgs = len(pathInpImgs)
    for ipath, pathInpImg in enumerate(pathInpImgsAbs):
        pathInpMsk = pathInpMasks[ipath]
        relPath = pathInpImgs[ipath]
        timgInp = skio.imread(pathInpImg)[:,:,:3]
        tmskInp = skio.imread(pathInpMsk)[:,:,3]
        #
        timgR = resizeToMaxSize(timgInp, outSize, porder=1)
        tmskR = resizeToMaxSize(tmskInp, outSize, porder=0)
        timgMod = np.dstack((timgR, tmskR))
        #
        if isAddBorder:
            timgMod = addBoundaries(timgMod, outShape, isDebug=isDebug)
        #
        todir = os.path.join(dirOut, os.path.dirname(relPath))
        foutImg = os.path.join(todir, '{0}-msk.png-channel.png'.format(os.path.basename(relPath)))
        make_dirs(todir)
        skio.imsave(foutImg, timgMod)
        print ('[{0}/{1}] -> ({2})'.format(ipath, numImgs, foutImg))
    fidxOut = os.path.join(dirOut, os.path.basename(fidxInp))
    print ('::copy Index file: [{0}]'.format(fidxOut))
    shutil.copy(fidxInp, fidxOut)
