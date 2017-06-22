#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import time
import shutil
import os
import sys
import math
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.transform as sktf
import skimage.exposure as skexp
import numpy as np
import keras.losses
import keras.callbacks as kall
import pandas as pd

from run01_CNN_Cls_Cervix_Only_train_v1 import preprocImgForInference, prepareCervixInfo

#####################################################
if __name__ == '__main__':
    numCls = 3
    batchSizeInference = 64
    imgSize = 224
    # (1) Setup Tran/Validation data
    fidxTest = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/02_test-x512-bordered/idx.txt'
    pathModelRestart = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/01_train_add-x512-original-bordered_Results/idx.txt_fold0_trn.csv_model_CNNCLS_EXT2_valLoss_v1.h5'
    if len(sys.argv)>1:
        fidxTest = sys.argv[1]
    if len(sys.argv)>2:
        pathModelRestart = sys.argv[2]
    if not os.path.isfile(fidxTest):
        raise Exception('*** ERROR *** Cant find test-index file! [{0}]'.format(fidxTest))
    if not os.path.isfile(pathModelRestart):
        raise Exception('*** ERROR *** Cant find model file! [{0}]'.format(pathModelRestart))
    #
    # pref = time.strftime('%Y.%m.%d-%H.%M.%S')
    pref = os.path.basename(os.path.splitext(pathModelRestart)[0])
    fidxOut = '{0}-results-{1}.csv'.format(fidxTest, pref)
    print (':: Result will be saved to: [{0}]'.format(fidxOut))
    wdirTest = os.path.dirname(fidxTest)
    #
    pathImgs = pd.read_csv(fidxTest)['path'].as_matrix()
    pathImgs = np.array([os.path.join(wdirTest, xx) for xx in pathImgs])
    #
    model = keras.models.load_model(pathModelRestart)
    model.summary()
    # (5) Preload data
    numVal = len(pd.read_csv(fidxTest))
    #
    arrResults = np.zeros((numVal,numCls))
    arrImgIdx = []
    for ipath, path in enumerate(pathImgs):
        arrImgIdx.append(os.path.basename(path))
        fimgMasked = '{0}-automasked.png'.format(path)
        timg = skio.imread(fimgMasked)
        tinf = prepareCervixInfo(timg)
        dataBatch = preprocImgForInference(timg, tinf,batchSize=batchSizeInference, isRandomize=True, imsize=imgSize)
        ret = model.predict_on_batch(dataBatch)
        arrResults[ipath] = np.mean(ret,axis=0)
        print ('\t[{0}/{1}] :\t {2}'.format(ipath, numVal, arrResults[ipath]))
    print ('---')
    with open(fidxOut, 'w') as f:
        f.write('image_name,Type_1,Type_2,Type_3\n')
        for ii in range(len(arrImgIdx)):
            fimgIdx = arrImgIdx[ii]
            probs = arrResults[ii]
            f.write('{0},{1:0.5f},{2:0.5f},{3:0.5f}\n'.format(fimgIdx, probs[0], probs[1], probs[2]))
    print ('-')

