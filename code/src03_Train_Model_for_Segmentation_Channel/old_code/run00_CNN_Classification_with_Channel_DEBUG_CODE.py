#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import cv2
import time
import shutil
import os
import math
from scipy import ndimage
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.transform as sktf
import skimage.morphology as skmorph
import skimage.exposure as skexp
import numpy as np
import keras
from keras.layers import Conv2D, UpSampling2D, \
    Flatten, Activation, Reshape, MaxPooling2D, Input, Dense, merge
from keras.models import Model
import keras.losses
import keras.callbacks as kall
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model as kplot
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

#####################################################
def buildModelCNN_Classification(inpShape=(256, 256, 3),
                                 numCls=3, kernelSize=3, numFlt = 16,
                                 numConv=2, numSubsampling=5, ppadding='valid', numHidden=None):
    fsiz = (kernelSize, kernelSize)
    psiz = (2, 2)
    dataInput = Input(shape=inpShape)
    #
    x = dataInput
    # (1) Conv-layers
    for cc in range(numSubsampling):
        for ii in range(numConv):
            x = Conv2D(filters=numFlt * (2 **cc), kernel_size=fsiz, activation='relu', padding=ppadding)(x)
        x = MaxPooling2D(pool_size=psiz, padding=ppadding)(x)
    # (2) flatening
    x = Flatten()(x)
    # (3) hidden dense-layers
    if numHidden is not None:
        if isinstance(numHidden, list):
            for numUnits in numHidden:
                x = Dense(units=numUnits, activation='relu')(x)

        else:
            x = Dense(units=numHidden, activation='relu')(x)
    # (4) multiclass-output
    x = Dense(units=numCls, activation='softmax')(x)
    retModel = Model(inputs=dataInput, outputs=x)
    return retModel

#####################################################
def readDataAsList(pidx, wdir=None, numCls=2):
    if wdir is None:
        wdir = os.path.dirname(pidx)
    lstpath = pd.read_csv(pidx)['path'].as_matrix()
    lstpath = [os.path.join(wdir,xx) for xx in lstpath]
    numPath = len(lstpath)
    dataX = []
    dataY = []
    print (':: readDataAsList()')
    for ii,pp in enumerate(lstpath):
        img4 = skio.imread(pp)
        img = (img4[:, :, :3].astype(np.float32) / 127.5) - 1.0
        if img4.ndim>2:
            msk = (img4[:, :, -1]>200).astype(np.float32)
            msk = np_utils.to_categorical(msk.reshape(-1), numCls)
            msk = msk.reshape(list(img.shape[:2]) + [numCls])
        else:
            msk = None
        dataX.append(img)
        dataY.append(msk)
        if (ii%100)==0:
            print ('\t[%d/%d] ...' % (ii, numPath))
    return (dataX, dataY, lstpath)

def readDataImagesCls(pidx, wdir=None, maxNum=None):
    if wdir is None:
        wdir = os.path.dirname(pidx)
    tdata = pd.read_csv(pidx)
    if maxNum is not None:
        numData = len(tdata)
        if maxNum>numData:
            maxNum = numData
        tdata = tdata[:maxNum]
    #
    dataY = tdata['type'].as_matrix() - 1
    tnumCls = len(np.unique(dataY))
    dataY   = np_utils.to_categorical(dataY, tnumCls)
    lstpath = tdata['path'].as_matrix()
    lstpath = [os.path.join(wdir, xx) for xx in lstpath]
    dataP = lstpath
    numPath = len(lstpath)
    dataX = None
    print (':: read images into memory...')
    for ipath, path in enumerate(lstpath):
        timg = skio.imread(path)
        if dataX is None:
            dataX = np.zeros([numPath] + list(timg.shape), dtype=np.uint8)
        if (ipath%20)==0:
            print ('\t[{0}/{1}]'.format(ipath, numPath))
        dataX[ipath] = timg
    return dataX, dataY, dataP

#####################################################
def preproc_image(pimg):
    ndim = pimg.ndim
    trnd = np.random.randint(4)
    # trnd = 1
    timg = pimg[:, :, :3].copy()
    tmsk = pimg[:, :,  3].copy()
    ret = pimg.copy()
    if trnd == 0:
        timg = skexp.equalize_hist(timg.astype(np.uint8), mask=tmsk).astype(np.float32) * 255.
    elif trnd == 1:
        vrnd = 1.0 + 0.2 * ( np.random.rand() - 0.5)
        timg = skexp.adjust_gamma(timg, vrnd, 2.71828 / np.exp(vrnd))
    elif trnd > 1:
        rndVals = 2.0 * np.random.rand(ndim,2) - 1.0
        rndVals[:, 0] *= 30
        rndVals[:, 1] = 1.0 + 0.2 * rndVals[:, 1]
        for ii in range(ndim):
            timg[:,:,ii] = rndVals[ii,0] + rndVals[ii,1] * timg[:,:,ii]
    timg[timg < 0] = 0
    timg[timg > 255] = 255
    timg[tmsk < 1] = 0
    ret[:, :,:3] = timg.copy()
    ret[:, :, 3] = tmsk.copy()
    return ret

def train_generator_CHANNEL(dataImg, pdataGenerator, numCls=2, batchSize=64, numRandGenPerBatch=16, imsize = 256, isRandomize=True):
    imsizeD2 = imsize/2
    numImg   = dataImg.shape[0]
    shapeMsk = dataImg.shape[1:3]
    numRB = shapeMsk[0] - imsize - 1
    numCB = shapeMsk[1] - imsize - 1
    shapeMskCat = list(shapeMsk) + [numCls]
    imgIdx = list(range(numImg))
    dataGeneratorFlow = pdataGenerator.flow(dataImg, None, batch_size=numRandGenPerBatch)
    while True:
        if isRandomize:
            dataImgG = next(dataGeneratorFlow)
        else:
            rndIdx = np.random.permutation(imgIdx)[:numRandGenPerBatch]
            dataImgG = dataImg[rndIdx].copy()
        # shape of data-generator can be changed and not equal [batchSize] !!!
        numRandGenPerBatchReal = dataImgG.shape[0]
        #
        mskCervix  = (dataImgG[:, :, :, 3] > 100)
        mskChannel = (dataImgG[:, :, :, 3] == 128)
        #
        dataX = np.zeros((batchSize, imsize, imsize, 3))
        dataY = np.zeros((batchSize, imsize * imsize, numCls))
        numSampleData = int(math.ceil(float(batchSize) / numRandGenPerBatchReal))
        tcnt = 0
        tcntImg = 0
        # rndRR = np.random.randint(0, shapeMsk[0] - imsize - 1, batchSize)
        # rndCC = np.random.randint(0, shapeMsk[1] - imsize - 1, batchSize)
        rndAng = np.random.rand(batchSize) * 2. * math.pi
        rndCos = np.cos(rndAng)
        rndSin = np.sin(rndAng)
        rndRad = np.random.rand(batchSize)
        while tcnt < batchSize:
            timg = dataImgG[tcntImg][:, :, :3] / 127.5 - 1.0
            tmsk = np_utils.to_categorical((dataImgG[tcntImg][:, :, 3] == 128).reshape(-1), numCls).reshape(shapeMskCat)
            #
            tmskCervix  = mskCervix[tcntImg]
            tmskChannel = mskChannel[tcntImg]
            (PCervix, RCervix) = cv2.minEnclosingCircle(np.array(np.where(tmskCervix)).transpose())
            (PChannel, RChannel) = cv2.minEnclosingCircle(np.array(np.where(tmskChannel)).transpose())
            Rrnd = RChannel
            if RChannel>(0.5*RCervix):
                Rrnd = 0.5*RCervix
            #
            for ii in range(numSampleData):
                trad = rndRad[tcnt] * Rrnd
                r00 = int(PChannel[1] + trad * rndCos[tcnt] - imsizeD2)
                c00 = int(PChannel[0] + trad * rndSin[tcnt] - imsizeD2)
                #
                if r00<0:
                    r00 = 0
                if c00<0:
                    c00 = 0
                if r00 >= numRB:
                    r00 = numRB
                if c00 >= numCB:
                    c00 = numCB
                #
                dataX[tcnt] = timg[r00:r00 + imsize, c00:c00 + imsize, :].copy()
                dataY[tcnt] = tmsk[r00:r00 + imsize, c00:c00 + imsize, :].reshape(-1, numCls).copy()
                tcnt += 1
            tcntImg +=1
        # print ('-')
        yield (dataX, dataY)

def calcDistArr2Point(parr2d, pp2d):
    sizArr = parr2d.shape[0]
    ret = np.linalg.norm(parr2d - np.tile(pp2d, (sizArr,1)), axis=1)
    return ret

def buildImageWithRotScaleAroundCenter(pimg, pcnt, pangDec, pscale, pcropSize, isDebug=False):
    # (1) precalc parameters
    angRad = (np.pi / 180.) * pangDec
    cosa = np.cos(angRad)
    sina = np.sin(angRad)
    # (2) prepare separate affine transformation matrices
    matShiftB = np.array([[1., 0., -pcnt[0]], [0., 1., -pcnt[1]], [0., 0., 1.]])
    matRot = np.array([[cosa, sina, 0.], [-sina, cosa, 0.], [0., 0., 1.]])
    matShiftF = np.array([[1., 0., +pcnt[0]], [0., 1., +pcnt[1]], [0., 0., 1.]])
    matScale = np.array([[pscale, 0., 0.], [0., pscale, 0.], [0., 0., 1.]])
    matShiftCrop = np.array([[1., 0., pcropSize[0] / 2.], [0., 1., pcropSize[1] / 2.], [0., 0., 1.]])
    # matTotal_OCV = matShiftF.dot(matRot.dot(matScale.dot(matShiftB)))
    # (3) build total-matrix
    matTotal = matShiftCrop.dot(matRot.dot(matScale.dot(matShiftB)))
    if isDebug:
        print ('(1) mat-shift-backward = \n{0}'.format(matShiftB))
        print ('(2) mat-scale = \n{0}'.format(matScale))
        print ('(3) mat-rot = \n{0}'.format(matRot))
        print ('(4) mat-shift-forward = \n{0}'.format(matShiftF))
        print ('(5) mat-shift-crop = \n{0}'.format(matShiftCrop))
        print ('---\n(*) mat-total = \n{0}'.format(matTotal))
    # (4) warp image with total affine-transform
    imgRet = cv2.warpAffine(pimg, matTotal[:2, :], pcropSize, borderMode=cv2.BORDER_REPLICATE)
    return imgRet

def prepareCervixAndChannelInfo(pimg, pRelChnSize = 0.4, isDebug = False):
    # (1) prepare masks
    tmsk = pimg[:, :, 3]
    timg = pimg[:, :, :3]
    tmsk_chn = (tmsk == 128)
    tmsk_crv = (tmsk > 100)
    # rc - mean first-idx -> row, second-idx -> column, xy - mean first-idx -> column, second-idx -> row :)
    # (2) find channel cover-circle and center of this corcle
    rc_pts_channel = np.array(np.where(tmsk_chn)).transpose()
    (rc_channel_cnt, r_channel) = cv2.minEnclosingCircle(rc_pts_channel)
    dist_chn2cnt = calcDistArr2Point(rc_pts_channel, rc_channel_cnt)
    r_channel_good = rc_pts_channel[dist_chn2cnt < pRelChnSize * r_channel, :]
    #FIXME: Fill holes before this step!!!
    # (2) prepare cervix contour
    contour_crv = tmsk_crv & (~skmorph.erosion(tmsk_crv, skmorph.disk(1)))
    rc_crvContour = np.array(np.where(contour_crv)).transpose()
    dist_contour2cnt = calcDistArr2Point(rc_crvContour, rc_channel_cnt)
    r_cervix = np.min(dist_contour2cnt)
    # rcCrvRminArg = np.argmin(rcRContour)
    if r_cervix<r_channel:
        r_cervix = r_channel
    ret = {
        'r_crv': r_cervix,
        'r_chn': r_channel,
        'r_chn_good': r_channel_good,
        'cnt_chn': rc_channel_cnt,
        'rc_chn': rc_pts_channel.copy()
    }
    if isDebug:
        retSize = 256
        newScale =  float(retSize)/(2.*r_cervix + 2.)
        xy_channel_cnt = rc_channel_cnt[::-1]
        timg_crop = buildImageWithRotScaleAroundCenter(timg, xy_channel_cnt, 45., newScale, (retSize, retSize), isDebug=False)
        #
        plt.subplot(2, 2, 1)
        plt.imshow(tmsk_chn)
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_channel, edgecolor='r', fill=False))
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_cervix, edgecolor='g', fill=False))
        plt.plot(r_channel_good[:, 1], r_channel_good[:, 0], 'y.')
        plt.subplot(2, 2, 2)
        plt.imshow(tmsk_crv)
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_channel, edgecolor='r', fill=False))
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_cervix, edgecolor='g', fill=False))
        plt.subplot(2, 2, 3)
        plt.imshow(timg)
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_channel, edgecolor='r', fill=False))
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_cervix, edgecolor='g', fill=False))
        plt.subplot(2, 2, 4)
        plt.imshow(timg_crop)
        plt.show()
    return ret

#####################################################
if __name__ == '__main__':
    # (1) Setup Tran/Validation data
    fidxTrn = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/01-data-512x512/idx.txt-train.txt'
    fidxVal = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/01-data-512x512/idx.txt-val.txt'
    wdir = os.path.dirname(fidxTrn)
    #
    pathImgs = pd.read_csv(fidxTrn)['path'].as_matrix()
    pathImgs = np.array([os.path.join(wdir, xx) for xx in pathImgs])
    numData = 100
    newSize = (256,256)
    #
    model = buildModelCNN_Classification(numHidden=None)
    model.summary()

    # trnX, trnY, _ = readDataImagesCls(fidxTrn)
    valX, valY, _ = readDataImagesCls(fidxVal, maxNum=10)
    trnX, trnY = valX, valY

    numTrn = trnX.shape[0]
    for ii in range(numTrn):
        tret = prepareCervixAndChannelInfo(trnX[ii], isDebug=True)
        print (tret)


