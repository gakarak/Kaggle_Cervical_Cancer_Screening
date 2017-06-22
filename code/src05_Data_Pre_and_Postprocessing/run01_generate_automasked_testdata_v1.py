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
def buildModelFCNN_UpSampling2D_V2_CERVIX(inpShape=(384, 384, 3), numCls=2, numConv=2, kernelSize=3, numFlt=8, ppad='same', numSubsampling=6, isDebug=False):
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
def buildModelFCNN_UpSampling2D_CHANNEL(inpShape=(256, 256, 3), numCls=2, kernelSize=3, numFlt = 8):
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
        _, R_crv = get_max_blob_mask(mskCrv > 0.5)
        msizCRV = math.ceil(R_crv * 0.04)
        if msizCRV < 3:
            msizCRV = 3
        mskCrv_Blob2 = skmorph.closing(mskCrv > 0.5, skmorph.disk(msizCRV))
        mskCrv_Blob3, _ = get_max_blob_mask(mskCrv_Blob2 > 0)
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

##########################################################
if __name__ == '__main__':
    isAddBorder = True
    sizeSegm = 512
    # sizeOut = 1024
    sizeOut = 512
    segmShape = (sizeSegm, sizeSegm)
    outShape = (sizeOut, sizeOut)
    #
    dirModels = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/models'
    pathModelCervix = '{0}/model_fcn_cervix_valAcc_v2.h5'.format(dirModels)
    pathModelChannel = '{0}/model_fcn_CHANNEL_valLoss_v1.h5'.format(dirModels)
    #
    fidxInp = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/test/00_test_original/idx.txt'
    dirInp = os.path.dirname(fidxInp)
    dirOut = '{0}-{1}x{1}'.format(dirInp, sizeOut)
    if isAddBorder:
        dirOut = '{0}-bordered'.format(dirOut)
    make_dirs(dirOut)
    # (1) input image paths
    pathInpImgs = pd.read_csv(fidxInp)['path'].as_matrix()
    pathInpImgsAbs = np.array([os.path.join(dirInp, xx) for xx in pathInpImgs])
    #
    numClsSegm = 2
    dataShapeSegm = (sizeSegm, sizeSegm, 3)
    # (2) loading Cervix Model
    modelCervix = buildModelFCNN_UpSampling2D_V2_CERVIX(inpShape=dataShapeSegm, numCls=numClsSegm, numSubsampling=5, numFlt=8)
    modelCervix.load_weights(pathModelCervix)
    modelCervix.summary()
    # (3) loading Channel Model
    modelChannel = buildModelFCNN_UpSampling2D_CHANNEL(inpShape=dataShapeSegm, numCls=numClsSegm)
    modelChannel.load_weights(pathModelChannel)
    modelChannel.summary()
    numImgs = len(pathInpImgs)
    for ipath, pathInpImg in enumerate(pathInpImgsAbs):
        relPath = pathInpImgs[ipath]
        timgInp = skio.imread(pathInpImg)
        timgSegmU8 = addBoundaries(resizeToMaxSize(timgInp, sizeSegm), segmShape)
        timgSegmBatch = timgSegmU8.astype(np.float32)/127.5 - 1.0
        timgSegmBatch = timgSegmBatch.reshape([1] + list(timgSegmBatch.shape))
        #
        pmapCervix = modelCervix.predict_on_batch(timgSegmBatch)[0].reshape(list(segmShape) + [numClsSegm])[:,:,1]
        pmapChannel = modelChannel.predict_on_batch(timgSegmBatch)[0].reshape(list(segmShape) + [numClsSegm])[:,:,1]
        pmapCervix = (255.*pmapCervix).astype(np.uint8)
        pmapChannel = (255.*pmapChannel).astype(np.uint8)
        mskCrvAndChn = makeCervixAndChannelMask(pmapCervixU8=pmapCervix, pmapChannelU8=pmapChannel)
        #
        if sizeOut == sizeSegm:
            timgOut = timgSegmU8.copy()
            timgOut = np.dstack((timgOut, mskCrvAndChn))
        else:
            timgOut = resizeToMaxSize(timgInp, poutSize=sizeOut)
            timgOut = addBoundaries(timgOut, newShape=outShape)
            tmskOut = resizeToMaxSize(mskCrvAndChn, poutSize=sizeOut, porder=0)
            tmskOut = addBoundaries(tmskOut, newShape=outShape)
            timgOut = np.dstack((timgOut, tmskOut))
        todir = os.path.join(dirOut, os.path.dirname(relPath))
        foutImg = os.path.join(todir, '{0}-msk.png-channel.png'.format(os.path.basename(relPath)))
        make_dirs(todir)
        skio.imsave(foutImg, timgOut)
        print ('[{0}/{1}] -> ({2})'.format(ipath, numImgs, foutImg))
    fidxOut = os.path.join(dirOut, os.path.basename(fidxInp))
    print ('::copy Index file: [{0}]'.format(fidxOut))
    shutil.copy(fidxInp, fidxOut)