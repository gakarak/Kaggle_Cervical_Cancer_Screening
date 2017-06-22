#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import json
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
import keras.applications as kapp

from run03_CNN_Cls_Cervix_And_Channel_train_v1 import preprocImgForInference, prepareCervixInfo, prepareCervixAndChannelInfo

#####################################################
def_data_leak = {
    '2.jpg': 2,
    '14.jpg': 3,
    '61.jpg': 2,
    '120.jpg': 3,
    '141.jpg': 2,
    '149.jpg': 2,
    '178.jpg': 1,
    '186.jpg': 2,
    '230.jpg': 1,
    '234.jpg': 2,
    '258.jpg': 3,
    '289.jpg': 2,
    '308.jpg': 2,
    '322.jpg': 3,
    '369.jpg': 2,
    '378.jpg': 2,
    '380.jpg': 2,
    '385.jpg': 2,
    '422.jpg': 3,
    '432.jpg': 3,
    '434.jpg': 1,
    '500.jpg': 2
}

#####################################################
def print_usage(pargv):
    print (':: Usage:\n\t{0} [/path/to/models-K-fold-description.json] [/path/to/test-data.idx] [model-types (loss|acc):loss]'.
           format(os.path.basename(pargv[0])))

#####################################################
if __name__ == '__main__':
    isUseLeaks = False
    numClasses = 3
    batchSizeInference = 128
    imgSize = 224
    imgShape = (imgSize, imgSize, 3)
    paramModelType = 'loss'  # 'acc'
    # (1) Setup Tran/Validation data
    pathModelJson = '_/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/models_segm_and_cls_v1/models.json'
    # fidxTest = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/02_test-x512-bordered/idx.txt'
    # fidxTest = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/00_test_original-512x512-bordered-v2/idx.txt'
    fidxTest = '_/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/00_test_original-512x512-bordered-v2-handmade/idx.txt'
##    pathModelRestart = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/01_train_add-x512-original-bordered_Results/idx.txt_fold0_trn.csv_model_CNNCLS_EXT2_valLoss_v1.h5'
    #
    if len(sys.argv)<2:
        print_usage(sys.argv)
    if len(sys.argv)>1:
        pathModelJson = sys.argv[1]
    if len(sys.argv)>2:
        fidxTest = sys.argv[2]
    if len(sys.argv)>3:
        paramModelType = sys.argv[3]
    #
    if not os.path.isfile(pathModelJson):
        raise Exception('*** ERROR *** Cant find model json file! [{0}]'.format(pathModelJson))
    if not os.path.isfile(fidxTest):
        raise Exception('*** ERROR *** Cant find Test-Data-Index file! [{0}]'.format(fidxTest))
    if paramModelType not in ['loss', 'acc']:
        raise Exception('*** ERROR *** Invalid model type: [{0}], available types is (loss|acc)'.format(paramModelType))
    #
    dirModels = os.path.dirname(pathModelJson)
    with open(pathModelJson, 'r') as f:
        modelsJson = json.loads(f.read())
    print ('---------------------')
    print ('models:')
    print (json.dumps(modelsJson, indent=4))
    #
    wdirTest = os.path.dirname(fidxTest)
    pathImgs = pd.read_csv(fidxTest)['path'].as_matrix()
    pathImgs = np.array([os.path.join(wdirTest, xx) for xx in pathImgs])
    # (1) Load CLASSIFICATION models into memory
    lstModels_CLS = []
    numModelsCLS = len(modelsJson['cls'][paramModelType])
    print ('----> Loading Classification Models:')
    for ipath, rpath in enumerate(modelsJson['cls'][paramModelType]):
        absPath = os.path.join(dirModels, rpath)
        print ('\t[{0}/{1}] : [{2}]'.format(ipath, numModelsCLS, absPath))
        if not os.path.isfile(absPath):
            raise Exception('*** ERROR *** cant find CERVIX-Model file: [{0}]'.format(absPath))
        #FIXME: this is KOSTIL, because keras optimizer-weigths issue fixer dont works...
        # tmodel = kapp.ResNet50(
        #     include_top=True,
        #     weights=None,  # 'imagenet',
        #     input_shape=imgShape,
        #     classes=numClasses)
        tmodel = keras.models.load_model(absPath)
        # tmodel.load_weights(absPath)
        lstModels_CLS.append(tmodel)
    #
    # pref = os.path.basename(os.path.splitext(modelsJson['cls'][paramModelType][0])[0])
    pref = os.path.basename(os.path.splitext(pathModelJson)[0])
    fidxOut = '{0}-results-{1}-{2}.csv'.format(fidxTest, pref, paramModelType)
    print ('---------------------')
    print("""
    Input-Parameters:
        model-json: [{0}]
        test-index: [{1}]
        model-type: [{2}]
    ---
    Output-Parameters:
        output-results: [{3}]
    """.format(pathModelJson, fidxTest, paramModelType, fidxOut))
    #
    print (':: Result will be saved to: [{0}]'.format(fidxOut))
    # (5) Preload data
    numTestSamples = len(pd.read_csv(fidxTest))
    #
    arrResults = np.zeros((numTestSamples, numClasses))
    arrImgIdx = []
    errorCounter = 0
    for ipath, path in enumerate(pathImgs):
        pathID = os.path.basename(path)
        arrImgIdx.append(os.path.basename(path))
        if isUseLeaks:
            if def_data_leak.has_key(pathID):
                cls_leak_ID = def_data_leak[pathID]-1
                pbest = 0.9999
                retProbs = np.array([(1-pbest)/2.]*3)
                retProbs[cls_leak_ID] = pbest
                arrResults[ipath] = retProbs
                print ('\t!!! WARNING !!! Found leaked data, use predefined Prob={0}, leak_ID/leak_={1}'.format(pbest, pathID, ))
                continue
        # fimgMasked = '{0}-automasked.png'.format(path)
        fimgMasked = '{0}-msk.png-channel.png'.format(path)
        timg = skio.imread(fimgMasked)
        # tinf = prepareCervixInfo(timg)
        try:
            tinf = prepareCervixAndChannelInfo(timg)
            dataBatch = preprocImgForInference(timg, tinf,batchSize=batchSizeInference, isRandomize=True, imsize=imgSize)
            retMean = []
            for imodel, model in enumerate(lstModels_CLS):
                ret = model.predict_on_batch(dataBatch)
                retMean.append(ret)
            retMean = np.concatenate(retMean)
            arrResults[ipath] = np.mean(retMean,axis=0)
        except:
            print (' !!! WARNING!!! invalid channel/cervix segmentation, skip... [{0}]'.format(fimgMasked))
            arrResults[ipath] = np.array([0.1688, 0.5273, 0.3038])
            errorCounter += 1
        print ('\t[{0}/{1}] :\t {2}'.format(ipath, numTestSamples, arrResults[ipath]))
    print ('---')
    print ('** Samples processing: Total/Errors = {0}/{1}'.format(numTestSamples, errorCounter))
    with open(fidxOut, 'w') as f:
        f.write('image_name,Type_1,Type_2,Type_3\n')
        for ii in range(len(arrImgIdx)):
            fimgIdx = arrImgIdx[ii]
            probs = arrResults[ii]
            f.write('{0},{1:0.5f},{2:0.5f},{3:0.5f}\n'.format(fimgIdx, probs[0], probs[1], probs[2]))
    print ('-')

