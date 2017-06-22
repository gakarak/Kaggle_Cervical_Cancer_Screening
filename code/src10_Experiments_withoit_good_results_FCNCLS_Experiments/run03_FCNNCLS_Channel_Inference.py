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

from run01_fcncls_channel_train_v2 import readDataImagesCls, prepareCervixAndChannelInfo, buildModelFCNNCLS_UpSampling2D_V3

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#####################################################
if __name__ == '__main__':
    # (1) Setup Tran/Validation data
    fidx = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/00_test_original-1024x1024-bordered/idx.txt'
    # fidx = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/00_test_original-512x512-bordered/idx.txt'
    fidxOut = '{0}-fcncls-v5.csv'.format(fidx)
    wdirVal = os.path.dirname(fidx)
    pathImgs = pd.read_csv(fidx)['path'].as_matrix()
    pathImgs = np.array([os.path.join(wdirVal, '{0}'.format(xx)) for xx in pathImgs])
    # (2) Input/Output models
    sizeImg = 1024
    # sizeImg = 512
    pathModelRestart = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-original-1024x1024-bordered/models_test/model_fcncls_channel_V2_valAcc_v1.h5'
    # pathModelRestart = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-original-512x512-bordered/model_fcncls_channel_V2_valLoss_v1.h5'
    dataShape = (sizeImg, sizeImg, 3)
    mskShape = (sizeImg, sizeImg)
    numCls = 4
    model = buildModelFCNNCLS_UpSampling2D_V3(inpShape=dataShape, numCls=numCls)
    model.load_weights(pathModelRestart)
    model.summary()
    # (5) Preload data
    numImg = len(pathImgs)
    #
    arrResults = np.zeros((numImg, numCls-1))
    arrImgIdx = []
    for ipath, path in enumerate(pathImgs):
        arrImgIdx.append(os.path.basename(path))
        fimgMasked = '{0}-msk.png-channel.png'.format(path)
        timg4 = skio.imread(fimgMasked)
        # tinf = prepareCervixAndChannelInfo(timg)
        # dataBatch = preprocImgForInference(timg, tinf,batchSize=64, isRandomize=True)
        timg3 = timg4[:,:,:3].astype(np.float32)/127.5 - 1.0
        tmsk  = timg4[:,:, 3]
        ret = model.predict_on_batch(timg3.reshape([1] + list(timg3.shape)))[0]
        retProb = ret.reshape((sizeImg, sizeImg, numCls))
        retProbTypes = ret.copy()
        # retProbTypes[:,0] = 0
        retSortIdx = np.argsort(-retProbTypes, axis=1)[:,0].reshape(mskShape)
        retSortIdx[tmsk<100]=0

        # (1)
        if np.sum(retSortIdx>0)>7:
            retProbL = retProb.reshape([-1, numCls])
            retSortIdxL = retSortIdx.reshape(-1)
            tmpProbs = []
            for xx in range(3):
                tprobCls = np.sum(retProb[:,:,xx+1] * (retSortIdx==(xx+1)) )
                tmpProbs.append(tprobCls)
            tmpProbs = np.array(tmpProbs)
            tmpProbs /= tmpProbs.sum()
            tmpProbs[tmpProbs<0.1] = 0.1
            tmpProbs /= tmpProbs.sum()
            probCls = tmpProbs
        else:
            probCls = np.array([0.1688, 0.5273, 0.3038])

        # (2)
        # if np.sum(retSortIdx>0)>7:
        #     tmpProbs = np.array([float(np.sum(retSortIdx==(xx+1))) for xx in range(3)])
        #     tmpProbs /= tmpProbs.sum()
        #     tmpProbs[tmpProbs>0.9] = 0.9
        #     tmpProbs /= tmpProbs.sum()
        #     probCls = tmpProbs
        # else:
        #     probCls = np.array([0.1688, 0.5273, 0.3038])

        # plt.imshow(retSortIdx)
        # plt.show()
        #
        # (3)
        # tmskChn = (tmsk.reshape(-1)==128)
        # probOnChn = ret.copy()
        # probOnChn[~tmskChn] = 0
        #
        # if np.sum(probOnChn)>0.001:
        #     probCls = np.sum(probOnChn, axis=0)[1:]
        # else:
        #     probCls = np.array([0.1688,0.5273,0.3038])
        # probClsMean = probCls / np.sum(probCls)
        # probClsSMax = softmax(probClsMean)
        # arrResults[ipath] = probClsMean
        arrResults[ipath] = probCls
        print ('\t[{0}/{1}]'.format(ipath, numImg))
    print ('---')
    with open(fidxOut, 'w') as f:
        f.write('image_name,Type_1,Type_2,Type_3\n')
        for ii in range(len(arrImgIdx)):
            fimgIdx = arrImgIdx[ii]
            probs = arrResults[ii]
            f.write('{0},{1:0.5f},{2:0.5f},{3:0.5f}\n'.format(fimgIdx, probs[0], probs[1], probs[2]))
    print ('-')

