#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import sys
import json
import shutil
import os
import pandas as pd
import numpy as np
import skimage.io as skio
import skimage.transform as sktf
import matplotlib.pyplot as plt
import cv2
from keras.utils.vis_utils import plot_model as kplot

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
import math
import skimage.measure as skmes
from scipy.ndimage.morphology import binary_fill_holes
import skimage.morphology as skmorph

##########################################################
# Helper functions
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
# basic models: cervix & channel segmentation
##### (CERVIX) Segmentation
def buildModelFCNN_UpSampling2D_V2_CERVIX(inpShape=(384, 384, 3), numCls=2, numConv=2, kernelSize=3, numFlt=8, ppad='same', numSubsampling=5, isDebug=False):
    dataInput = Input(shape=inpShape)
    fsiz = (kernelSize, kernelSize)
    psiz = (2, 2)
    x = dataInput
    # -------- Encoder --------
    lstMaxPools = []
    for cc in range(numSubsampling):
        for ii in range(numConv):
            x = Conv2D(filters=numFlt * (2**cc), kernel_size=fsiz, padding=ppad, activation='relu')(x)
        lstMaxPools.append(x)
        x = MaxPooling2D(pool_size=psiz)(x)
    # -------- Decoder --------
    for cc in range(numSubsampling):
        for ii in range(numConv):
            x = Conv2D(filters=numFlt * (2 ** (numSubsampling - 1 -cc)), kernel_size=fsiz, padding=ppad, activation='relu')(x)
        x = UpSampling2D(size=psiz)(x)
        # if cc< (numSubsampling-1):
        #     x = keras.layers.concatenate([x, lstMaxPools[-1 - cc]], axis=-1)
        x = keras.layers.concatenate([x, lstMaxPools[-1 - cc]], axis=-1)
    #
    # 1x1 Convolution: emulation of Dense layer
    x = Conv2D(filters=numCls, kernel_size=(1,1), padding='valid', activation='linear')(x)
    x = Reshape([-1, numCls])(x)
    x = Activation('softmax')(x)
    retModel = Model(dataInput, x)
    if isDebug:
        retModel.summary()
        fimg_model = 'model_graph_FCNN_UpSampling2D_V2.png'
        kplot(retModel, fimg_model, show_shapes=True)
        plt.imshow(skio.imread(fimg_model))
        plt.show()
    return retModel

##### (CHANNEL) Segmentation
def buildModelFCNN_UpSampling2D_CHANNEL(inpShape=(256, 256, 3), numCls=2, kernelSize=3, numFlt = 12):
    dataInput = Input(shape=inpShape)
    # -------- Encoder --------
    # Conv #1
    conv1 = Conv2D(filters= numFlt * (2**0), kernel_size=(kernelSize,kernelSize),
                   padding='same', activation='relu')(dataInput)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    # Conv #2
    conv2 = Conv2D(filters= numFlt * (2**1), kernel_size=(kernelSize, kernelSize),
                   padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # Conv #3
    conv3 = Conv2D(filters= numFlt * (2**2), kernel_size=(kernelSize, kernelSize),
                   padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # Conv #4
    conv4 = Conv2D(filters= numFlt * (2**3), kernel_size=(kernelSize, kernelSize),
                   padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # Conv #5
    conv5 = Conv2D(filters= numFlt * (2**4), kernel_size=(kernelSize, kernelSize),
                   padding='same', activation='relu')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    #
    # -------- Decoder --------
    # UpConv #1
    upconv1 = Conv2D(filters= numFlt * (2**4), kernel_size=(kernelSize, kernelSize),
                     padding='same', activation='relu')(pool5)
    up1 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(upconv1),conv5], axis=-1)
    # UpConv #2
    upconv2 = Conv2D(filters= numFlt * (2**3), kernel_size=(kernelSize, kernelSize),
                     padding='same', activation='relu')(up1)
    up2 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(upconv2), conv4], axis=-1)
    # UpConv #3
    upconv3 = Conv2D(filters= numFlt * (2**2), kernel_size=(kernelSize, kernelSize),
                     padding='same', activation='relu')(up2)
    up3 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(upconv3), conv3], axis=-1)
    # UpConv #4
    upconv4 = Conv2D(filters= numFlt * (2**1), kernel_size=(kernelSize, kernelSize),
                     padding='same', activation='relu')(up3)
    up4 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(upconv4), conv2], axis=-1)
    # UpConv #5
    upconv5 = Conv2D(filters= numFlt * (2**0), kernel_size=(kernelSize, kernelSize),
                     padding='same', activation='relu')(up4)
    up5 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(upconv5), conv1], axis=-1)
    #
    # 1x1 Convolution: emulation of Dense layer
    convCls = Conv2D(filters=numCls, kernel_size=(1,1), padding='valid', activation='linear')(up5)
    # sizeReshape = np.prod(inpShape[:2])
    ret = Reshape([-1, numCls])(convCls)
    ret = Activation('softmax')(ret)
    retModel = Model(dataInput, ret)
    return retModel

##########################################################
def get_max_blob_mask(pmsk):
    tlbl = skmes.label(pmsk)
    tprops = skmes.regionprops(tlbl)
    arr_areas = np.array([xx.area for xx in tprops])
    idxMax = np.argmax(arr_areas)
    lblMax = tprops[idxMax].label
    retMsk = (tlbl == lblMax)
    (P0, R) = cv2.minEnclosingCircle(np.array(np.where(retMsk)).transpose())
    return (retMsk, R)

def makeCervixAndChannelMask(pmapCervixU8, pmapChannelU8):
    try:
        # (1) load images
        mskChn = pmapChannelU8.astype(np.float) / 255.
        mskCrv = pmapCervixU8.astype(np.float) / 255.
        # (2) preprocess Cervix Mask
        thresholdCrv = 0.5
        if np.sum(mskCrv>0.5)<50:
            thresholdCrv = 0.75 * np.max(mskCrv)
        _, R_crv = get_max_blob_mask(mskCrv > thresholdCrv)
        msizCRV = math.ceil(R_crv * 0.04)
        if msizCRV < 3:
            msizCRV = 3
        mskCrv_Blob2 = skmorph.closing(mskCrv > thresholdCrv, skmorph.disk(msizCRV))
        mskCrv_Blob3, _ = get_max_blob_mask(mskCrv_Blob2 > 0)
        mskCrv_Blob4 = binary_fill_holes(mskCrv_Blob3)
        # (3) preprocess Channel mask
        thresholdChn = 0.5
        if np.sum(mskChn>0.5)<50:
            thresholdChn = 0.75 * np.max(mskChn)
        mskChn_Blob1 = mskChn.copy()
        mskChn_Blob1[~mskCrv_Blob4] = 0
        # (3.1) check zero-channel-mask
        if np.sum(mskChn_Blob1 > thresholdChn) < 1:
            mskChn_Blob1 = skmorph.skeletonize(mskCrv_Blob4)
            mskChn_Blob1 = skmorph.closing(mskChn_Blob1, skmorph.disk(5))
            R_chn = 5
        else:
            _, R_chn = get_max_blob_mask(mskChn_Blob1 > 0.5)
        msizChn = math.ceil(R_chn * 0.1)
        if msizChn < 3:
            msizChn = 3
        mskChn_Blob2 = skmorph.closing(mskChn_Blob1 > 0.5, skmorph.disk(msizChn))
        mskChn_Blob2[~mskCrv_Blob4] = 0
        # (3.1) check zero-channel-mask
        if np.sum(mskChn_Blob2 > 0) < 1:
            mskChn_Blob2 = skmorph.skeletonize(mskCrv_Blob4)
            mskChn_Blob2 = skmorph.closing(mskChn_Blob2, skmorph.disk(5))
            #
        mskChn_Blob3, _ = get_max_blob_mask(mskChn_Blob2 > 0)
        mskChn_Blob4 = binary_fill_holes(mskChn_Blob3)
        # (4) Composing output mask
        mskShape = pmapCervixU8.shape[:2]
        mskOut = 64 * np.ones(mskShape, dtype=np.uint8)
        mskOut[mskCrv_Blob4] = 255
        mskOut[mskChn_Blob4] = 128
    except Exception as err:
        print ('\t!!! ERROR !!! [{0}]'.format(err))
        mskOut = np.zeros(pmapCervixU8.shape[:2], dtype=np.uint8)
    return mskOut

def print_usage(pargv):
    print (':: Usage:\n\t{0} [/path/to/models-K-fold-description.json] [/path/to/test-data.idx] [sizeOut:512] [model-types (loss|acc):loss] [isBordered (brd,non): brd]'.
           format(os.path.basename(pargv[0])))

##########################################################
if __name__ == '__main__':
    isAddBorder = True
    sizeSegm = 512
    # paramSizeOut = 1024
    paramSizeOut = 512
    numClsSegm = 2
    paramModelType = 'loss' # 'acc'
    #
    pathModelJson = '_/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/01_train_add-x512-original-bordered_Results/models_bk2_3fold/models.json'
    fidxTest = '_/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/00_test_original/idx.txt'
    if len(sys.argv)<2:
        print_usage(sys.argv)
    if len(sys.argv)>1:
        pathModelJson = sys.argv[1]
    if len(sys.argv)>2:
        fidxTest = sys.argv[2]
    if len(sys.argv)>3:
        paramSizeOut = int(sys.argv[3])
    if len(sys.argv)>4:
        paramModelType = sys.argv[4]
    if len(sys.argv)>5:
        tmpBrd = sys.argv[5]
        if tmpBrd == 'brd':
            isAddBorder = True
        else:
            isAddBorder = False
    #
    segmShape = (sizeSegm, sizeSegm)
    dataShapeSegm = (sizeSegm, sizeSegm, 3)
    outShape = (paramSizeOut, paramSizeOut)
    dirInp = os.path.dirname(fidxTest)
    dirOut = '{0}-{1}x{1}-{2}'.format(dirInp, paramSizeOut, paramModelType)
    if isAddBorder:
        dirOut = '{0}-bordered'.format(dirOut)
    print("""
    Input-Parameters:
        model-json: [{0}]
        test-index: [{1}]
        model-type: [{2}]
    ---
    Output-Parameters:
        is-bordered:      {3}
        output-directory: [{4}]
        output-shape:     {5}
    """.format(pathModelJson, fidxTest, paramModelType, isAddBorder, dirOut, outShape))
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
    # (1.1) Load CERVIX models into memory
    lstModels_Cervix = []
    numModelsCervix = len(modelsJson['cervix'][paramModelType])
    print ('----> Loading CERVIX Models:')
    for ipath, rpath in enumerate(modelsJson['cervix'][paramModelType]):
        absPath = os.path.join(dirModels, rpath)
        if not os.path.isfile(absPath):
            raise Exception('*** ERROR *** cant find CERVIX-Model file: [{0}]'.format(absPath))
        tmodel = buildModelFCNN_UpSampling2D_V2_CERVIX(inpShape=dataShapeSegm)
        tmodel.load_weights(absPath)
        lstModels_Cervix.append(tmodel)
        print ('\t[{0}/{1}]'.format(ipath, numModelsCervix))
    # (1.2) Load CHANNEL models into memory
    lstModels_Channel = []
    numModelsCervix = len(modelsJson['channel'][paramModelType])
    print ('----> Loading CHANNEL Models:')
    for ipath, rpath in enumerate(modelsJson['channel'][paramModelType]):
        absPath = os.path.join(dirModels, rpath)
        if not os.path.isfile(absPath):
            raise Exception('*** ERROR *** cant find CHANNEL-Model file: [{0}]'.format(absPath))
        tmodel = buildModelFCNN_UpSampling2D_CHANNEL(inpShape=dataShapeSegm)
        tmodel.load_weights(absPath)
        lstModels_Channel.append(tmodel)
        print ('\t[{0}/{1}]'.format(ipath, numModelsCervix))
    #
    make_dirs(dirOut)
    # (2) input image paths
    pathInpImgs = pd.read_csv(fidxTest)['path'].as_matrix()
    pathInpImgsAbs = np.array([os.path.join(dirInp, xx) for xx in pathInpImgs])
    # (3) Iterate over images
    numImgs = len(pathInpImgs)
    for ipath, pathInpImg in enumerate(pathInpImgsAbs):
        relPath = pathInpImgs[ipath]
        timgInp = skio.imread(pathInpImg)
        timgSegmU8 = addBoundaries(resizeToMaxSize(timgInp, sizeSegm), segmShape)
        timgSegmBatch = timgSegmU8.astype(np.float32)/127.5 - 1.0
        timgSegmBatch = timgSegmBatch.reshape([1] + list(timgSegmBatch.shape))
        # (3.1) cervix-iterations
        pmapCervix = None
        for tmodelCervix in lstModels_Cervix:
            if pmapCervix is None:
                pmapCervix  = tmodelCervix.predict_on_batch(timgSegmBatch)[0].reshape(list(segmShape) + [numClsSegm])[:,:,1]
            else:
                pmapCervix += tmodelCervix.predict_on_batch(timgSegmBatch)[0].reshape(list(segmShape) + [numClsSegm])[:,:,1]
        pmapCervix /= len(lstModels_Cervix)
        # (3.2) channel iterations
        pmapChannel = None
        for tmodelChannel in lstModels_Channel:
            if pmapChannel is None:
                pmapChannel  = tmodelChannel.predict_on_batch(timgSegmBatch)[0].reshape(list(segmShape) + [numClsSegm])[:,:,1]
            else:
                pmapChannel += tmodelChannel.predict_on_batch(timgSegmBatch)[0].reshape(list(segmShape) + [numClsSegm])[:,:,1]
        pmapChannel /= len(lstModels_Channel)
        #
        pmapCervix = (255.*pmapCervix).astype(np.uint8)
        pmapChannel = (255.*pmapChannel).astype(np.uint8)
        mskCrvAndChn = makeCervixAndChannelMask(pmapCervixU8=pmapCervix, pmapChannelU8=pmapChannel)
        #
        if paramSizeOut == sizeSegm:
            timgOut = timgSegmU8.copy()
            timgOut = np.dstack((timgOut, mskCrvAndChn))
        else:
            timgOut = resizeToMaxSize(timgInp, poutSize=paramSizeOut)
            timgOut = addBoundaries(timgOut, newShape=outShape)
            tmskOut = resizeToMaxSize(mskCrvAndChn, poutSize=paramSizeOut, porder=0)
            tmskOut = addBoundaries(tmskOut, newShape=outShape)
            timgOut = np.dstack((timgOut, tmskOut))
        todir = os.path.join(dirOut, os.path.dirname(relPath))
        foutImg = os.path.join(todir, '{0}-msk.png-channel.png'.format(os.path.basename(relPath)))
        make_dirs(todir)
        skio.imsave(foutImg, timgOut)
        print ('[{0}/{1}] -> ({2})'.format(ipath, numImgs, foutImg))
    fidxOut = os.path.join(dirOut, os.path.basename(fidxTest))
    print ('::copy Index file: [{0}]'.format(fidxOut))
    shutil.copy(fidxTest, fidxOut)