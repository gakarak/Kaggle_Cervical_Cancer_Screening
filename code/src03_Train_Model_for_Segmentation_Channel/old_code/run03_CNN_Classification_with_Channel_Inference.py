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

from run01_CNN_Classification_with_Channel_train import readDataImagesCls,buildModelCNN_Classification,\
    preprocImgForInference, prepareCervixAndChannelInfo

#####################################################
if __name__ == '__main__':
    # (1) Setup Tran/Validation data
    # fidxTest = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/02_test-x512-bordered/idx.txt'
    fidxTest = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/00_test_original-512x512-bordered-v2/idx.txt'
    # wdirTrn = os.path.dirname(fidxTrn)
    wdirVal = os.path.dirname(fidxTest)
    #
    pathImgs = pd.read_csv(fidxTest)['path'].as_matrix()
    pathImgs = np.array([os.path.join(wdirVal, xx) for xx in pathImgs])
    # (2) Input/Output models
    pathModelRestart = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/01-data-512x512/backup_training_cls/model_CNN_Classification_valAcc_v1.h5'
    # pathModelRestart = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/01-data-512x512/backup_training_cls/model_CNN_Classification_valLoss_v1.h5'
    #
    prefResult = os.path.basename(os.path.splitext(pathModelRestart)[0])
    fidxOut = '{0}-results-{1}.csv'.format(fidxTest, prefResult)
    dataShape = (512, 512, 3)
    numCls = 3
    # model = buildModelFCNN_UpSampling2D(inpShape=dataShape, numCls=numCls)
    # model.load_weights(pathModelRestart)
    model = keras.models.load_model(pathModelRestart)
    model.summary()
    # (5) Preload data
    numVal = len(pd.read_csv(fidxTest))
    #
    arrResults = np.zeros((numVal,numCls))
    arrImgIdx = []
    for ipath, path in enumerate(pathImgs):
        arrImgIdx.append(os.path.basename(path))
        # fimgMasked = '{0}-automasked.png'.format(path)
        fimgMasked = '{0}-msk.png-channel.png'.format(path)
        timg = skio.imread(fimgMasked)
        try:
            tinf = prepareCervixAndChannelInfo(timg)
            dataBatch = preprocImgForInference(timg, tinf,batchSize=512, isRandomize=True)
            ret = model.predict_on_batch(dataBatch)
            arrResults[ipath] = np.mean(ret, axis=0)
        except:
            arrResults[ipath] = np.array([0.1688, 0.5273, 0.3038])
        print ('\t[{0}/{1}]'.format(ipath, numVal))
    print ('---')
    with open(fidxOut, 'w') as f:
        f.write('image_name,Type_1,Type_2,Type_3\n')
        for ii in range(len(arrImgIdx)):
            fimgIdx = arrImgIdx[ii]
            probs = arrResults[ii]
            f.write('{0},{1:0.5f},{2:0.5f},{3:0.5f}\n'.format(fimgIdx, probs[0], probs[1], probs[2]))
    print ('-')

