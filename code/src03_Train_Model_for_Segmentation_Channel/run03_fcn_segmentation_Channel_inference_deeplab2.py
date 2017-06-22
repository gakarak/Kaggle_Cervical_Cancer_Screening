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

from run01_fcn_segmentation_Channel_train import readDataVal, readDataAsList, buildModelFCNN_UpSampling2D

#####################################################
if __name__ == '__main__':
    isDebug = False
    # (1) Setup Tran/Validation data
    # fidxTrn = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/01-data-512x512/idx.txt-train.txt'
    # fidxVal = '/mnt/data6T/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/02_test-x512-bordered/idx.txt'
    fidxVal = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/01_train_add-x512-original-bordered/idx.txt'
    # wdirTrn = os.path.dirname(fidxTrn)
    wdirVal = os.path.dirname(fidxVal)
    #
    pathImgs = pd.read_csv(fidxVal)['path'].as_matrix()
    pathImgs = np.array([os.path.join(wdirVal, xx) for xx in pathImgs])
    # (2) Input/Output models
    # pathModelVal = '{0}/model_fcn_CHANNEL_vsl_v1.h5'.format(wdirTrn)
    # pathModelValAcc = '{0}/model_fcn_CHANNEL_acc_v1.h5'.format(wdirTrn)
    # pathModelLatest = '{0}/model_fcn_CHANNEL_latest_v1.h5'.format(wdirTrn)
    # pathLog = '%s-log.csv' % pathModelVal
    # (4) Load existing model
    # pathModelRestart = pathModelVal
    pathModelRestart = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/10_models/model_fcn_CHANNEL_valLoss_v1.h5'
    dataShape = (512, 512, 3)
    numCls = 2
    model = buildModelFCNN_UpSampling2D(inpShape=dataShape, numCls=numCls)
    model.load_weights(pathModelRestart)
    model.summary()
    # (5) Preload data
    numVal = len(pd.read_csv(fidxVal))
    #
    plt.figure(figsize=(20,10))
    for ipath, path in enumerate(pathImgs):
        foutPathFig = '{0}-debug-figure-CHANNEL.png'.format(path)
        foutPathImg = '{0}-msk-CHANNEL.png'.format(path)
        timg0 = skio.imread(path).astype(np.float32)
        shapeMsk = list(timg0.shape[:2]) + [numCls]
        timg = timg0/127.5 - 1.0
        timg = timg.reshape([1] + list(timg.shape))
        tret = model.predict_on_batch(timg)
        tret = tret.reshape([tret.shape[0]] + shapeMsk)
        msku8 = (255. * tret[0][:,:,1]).astype(np.uint8)
        timgOut = np.dstack( (timg0, msku8) ).astype(np.uint8)
        skio.imsave(foutPathImg, timgOut)
        if isDebug:
            plt.subplot(1, 3, 1)
            plt.title('image')
            plt.imshow(timg0.astype(np.uint8))
            plt.subplot(1, 3, 2)
            plt.title('mask')
            plt.imshow(tret[0][:,:,1])
            plt.subplot(1, 3, 3)
            plt.imshow(timgOut)
            plt.title('masked')
            plt.savefig(foutPathFig)
            plt.show()
        if (ipath%10)==0:
            print ('\t[{0}/{1}]'.format(ipath, numVal))
        # print ('---')

