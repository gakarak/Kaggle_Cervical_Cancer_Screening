#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import time
import shutil
import os
import math
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.transform as sktf
import skimage.exposure as skexp
import numpy as np
import keras
from keras.layers import Conv2D, UpSampling2D, \
    Flatten, Activation, Reshape, MaxPooling2D, Input, merge
from keras.models import Model
import keras.losses
import keras.callbacks as kall
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model as kplot
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

from run01_fcn_segmentation_cervix_train_deeplab2 import readDataVal, readDataAsList, buildModelFCNN_UpSampling2D, buildModelFCNN_UpSampling2D_V2

#####################################################
if __name__ == '__main__':
    isDebug = True
    # (1) Setup Tran/Validation data
    # fidxTrn = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/01-data-512x512/idx.txt-train.txt'
    fidxTrn = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/01_train_add-x512-original-bordered_Results/idx.txt-train.txt'
    fidxVal = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/02_test-x512-bordered/idx.txt'
    wdirTrn = os.path.dirname(fidxTrn)
    wdirVal = os.path.dirname(fidxVal)
    #
    pathImgs = pd.read_csv(fidxVal)['path'].as_matrix()
    pathImgs = np.array([os.path.join(wdirVal, xx) for xx in pathImgs])
    # (2) Input/Output models
    pathModelPrefix = '{0}/model_fcn_cervix_EXT1'.format(wdirTrn)
    pathModelValLoss = '{0}_valLoss_v2.h5'.format(pathModelPrefix)
    pathModelValAcc = '{0}_valAcc_v2.h5'.format(pathModelPrefix)
    pathModelLatest = '{0}_Latest_v2.h5'.format(pathModelPrefix)
    pathLog = '%s-log.csv' % pathModelValLoss
    # (4) Load existing model
    pathModelRestart = pathModelValLoss
    dataShape = (512, 512, 3)
    numCls = 2
    # model = buildModelFCNN_UpSampling2D(inpShape=dataShape, numCls=numCls)
    model = buildModelFCNN_UpSampling2D_V2(inpShape=dataShape, numCls=numCls, numSubsampling=5, numFlt=8)
    model.load_weights(pathModelRestart)
    # (5) Preload data
    numVal = len(pd.read_csv(fidxVal))
    #
    for ipath, path in enumerate(pathImgs):
        foutPathFig = '{0}-debug-figure-cervix.png'.format(path)
        foutPathImg = '{0}-msk-cervix.png'.format(path)
        timg0 = skio.imread(path).astype(np.float32)
        shapeMsk = list(timg0.shape[:2]) + [numCls]
        timg = timg0/127.5 - 1.0
        timg = timg.reshape([1] + list(timg.shape))
        tret = model.predict_on_batch(timg)
        tret = tret.reshape([tret.shape[0]] + shapeMsk)
        msku8 = (255. * tret[0][:,:,1]).astype(np.uint8)
        timgOut = np.dstack( (timg0, msku8) ).astype(np.uint8)
        #
        if isDebug:
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.title('image')
            plt.imshow(timg0.astype(np.uint8))
            plt.subplot(1, 3, 2)
            plt.title('mask')
            plt.imshow(tret[0][:,:,1])
            plt.subplot(1, 3, 3)
            plt.imshow(timgOut)
            plt.title('masked')
            plt.show()
        else:
            # plt.savefig(foutPathFig)
            skio.imsave(foutPathImg, timgOut)
        if (ipath%10)==0:
            print ('\t[{0}/{1}]'.format(ipath, numVal))
        # print ('---')

