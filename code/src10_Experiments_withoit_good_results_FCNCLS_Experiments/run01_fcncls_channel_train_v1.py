#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import cv2
import time
import shutil
import os
import sys
import math
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.morphology as skmorph
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
import keras.optimizers as kopt

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model as kplot
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

#####################################################
def buildModelFCNNCLS_UpSampling2D_V3(inpShape=(256, 256, 3),
                                      numCls=4,
                                      numConv=2, kernelSize=3, numFlt=8,
                                      isUNetStyle=True,
                                      unetStartLayer=1,
                                      ppad='same', numSubsampling=6, isDebug=False):
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
        if isUNetStyle:
            if cc<(numSubsampling-unetStartLayer):
                x = keras.layers.concatenate([x, lstMaxPools[-1 - cc]], axis=-1)
    #
    # 1x1 Convolution: emulation of Dense layer
    x = Conv2D(filters=numCls, kernel_size=(1,1), padding='valid', activation='linear')(x)
    x = Reshape([-1, numCls])(x)
    x = Activation('softmax')(x)
    retModel = Model(dataInput, x)
    if isDebug:
        retModel.summary()
        fimg_model = 'model_graph_FCNN_UpSampling2D_V3.png'
        kplot(retModel, fimg_model, show_shapes=True)
        plt.imshow(skio.imread(fimg_model))
        plt.show()
    return retModel

#####################################################
def preproc_image(pimg):
    ndim = pimg.ndim
    # trnd = np.random.randint(4)
    trnd = 1
    timg = pimg[:, :, :3].copy()
    ret = pimg.copy()
    if trnd == 0:
        timg = skexp.equalize_hist(timg.astype(np.uint8)).astype(np.float32) * 255.
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
    ret[:, :,:3] = timg.copy()
    return ret

#####################################################
def readDataImagesCls(pidx, wdir=None, maxNum=None):
    if wdir is None:
        wdir = os.path.dirname(pidx)
    tdata = pd.read_csv(pidx)
    if maxNum is not None:
        numData = len(tdata)
        if maxNum > numData:
            maxNum = numData
        tdata = tdata[:maxNum]
    #
    dataCls = tdata['type'].as_matrix()
    lstpath = tdata['path'].as_matrix()
    lstpath = [os.path.join(wdir, xx) for xx in lstpath]
    dataPaths = lstpath
    numPath = len(lstpath)
    dataX = None
    print (':: read images into memory...')
    for ipath, path in enumerate(lstpath):
        timg = skio.imread(path)
        if dataX is None:
            dataX = np.zeros([numPath] + list(timg.shape), dtype=np.uint8)
        if (ipath % 20) == 0:
            print ('\t[{0}/{1}]'.format(ipath, numPath))
        dataX[ipath] = timg
    return dataX, dataCls, dataPaths

#####################################################
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
    imgRet = cv2.warpAffine(pimg, matTotal[:2, :], pcropSize, flags=cv2.INTER_NEAREST)
    return imgRet

def prepareCervixAndChannelInfo(pimg, pRelChnSize = 0.7, isDebug = False):
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
    r_cervix_min = np.min(dist_contour2cnt)
    r_cervix_max = np.max(dist_contour2cnt)
    # rcCrvRminArg = np.argmin(rcRContour)
    if r_cervix_min<r_channel:
        r_cervix_min = r_channel
    ret = {
        'r_crv_min': r_cervix_min,
        'r_crv_max': r_cervix_max,
        'r_chn': r_channel,
        'r_chn_good': pRelChnSize * r_channel,
        'cnt_chn': rc_channel_cnt,
        'rc_chn': r_channel_good.copy()
    }
    if isDebug:
        retSize = 256
        newScale =  float(retSize)/(2.*r_cervix_min + 2.)
        xy_channel_cnt = rc_channel_cnt[::-1]
        timg_crop = buildImageWithRotScaleAroundCenter(timg, xy_channel_cnt, 45., newScale, (retSize, retSize), isDebug=False)
        #
        plt.subplot(2, 2, 1)
        plt.imshow(tmsk_chn)
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_channel, edgecolor='r', fill=False))
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_cervix_min, edgecolor='g', fill=False))
        plt.plot(r_channel_good[:, 1], r_channel_good[:, 0], 'y.')
        plt.subplot(2, 2, 2)
        plt.imshow(tmsk_crv)
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_channel, edgecolor='r', fill=False))
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_cervix_min, edgecolor='g', fill=False))
        plt.subplot(2, 2, 3)
        plt.imshow(timg)
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_channel, edgecolor='r', fill=False))
        plt.gcf().gca().add_artist(plt.Circle(rc_channel_cnt[::-1], r_cervix_min, edgecolor='g', fill=False))
        plt.subplot(2, 2, 4)
        plt.imshow(timg_crop)
        plt.show()
    return ret

def buildImgInfoList(dataImg):
    numImg = dataImg.shape[0]
    print (":: Prepare image info ({0})".format(dataImg.shape))
    ret = []
    for ii in range(numImg):
        timg = dataImg[ii]
        tinfo = prepareCervixAndChannelInfo(timg)
        ret.append(tinfo)
        if (ii%10)==0:
            print ('\t[{0}/{1}]'.format(ii, numImg))
    return ret

#####################################################
def getRandomInRange(vrange, pnum=None):
    vmin,vmax = vrange
    if pnum is None:
        trnd = np.random.rand()
    else:
        trnd = np.random.rand(pnum)
    ret = vmin + (vmax-vmin)*trnd
    return ret

#####################################################
def train_generator(dataImg, dataCls, dataInfo, numCls=4, batchSize=64, imsize = 256, isRandomize=True, angleRange=(-24.,+24.), isDebug=False):
    numImg   = dataImg.shape[0]
    shapeMsk = dataImg.shape[1:3]
    # #Cls is equal number of channel types + background
    cropSize = (imsize, imsize)
    while True:
        rndIdx = np.random.randint(0, numImg, batchSize)
        dataX = np.zeros((batchSize, imsize, imsize, 3))
        dataY = np.zeros((batchSize, imsize * imsize, numCls))
        rndIsFlip = np.random.rand(batchSize)>0.5
        rndShiftMean = 0.2 * getRandomInRange((-1., 1.0), pnum=batchSize)
        rndShiftStd = 1.0 + 0.2 * getRandomInRange((-1.0, 1.0), pnum=batchSize)
        for ii, idx in enumerate(rndIdx):
            timg4 = dataImg[idx]
            tinf = dataInfo[idx]
            tcls = dataCls[idx]
            # (1) get random point around channel
            PTS_chn_rc = tinf['rc_chn']
            rndChnPos = np.random.randint(PTS_chn_rc.shape[0])
            Pxy = PTS_chn_rc[rndChnPos][::-1]
            #
            Rchn0 = tinf['r_chn']
            Rcrv1 = tinf['r_crv_min']
            Rcrv2 = tinf['r_crv_max']
            #
            Rmin = Rchn0 * 3
            if Rmin>Rcrv1:
                Rmin = Rcrv1
            Rmax = Rcrv2
            if Rcrv2>2*Rcrv1:
                Rmax = 2*Rcrv1
            Rrnd = getRandomInRange((Rmin, Rmax))
            Arnd = getRandomInRange(angleRange)
            cropScale = float(imsize)/Rrnd
            timgCrop = buildImageWithRotScaleAroundCenter(timg4, Pxy, Arnd, cropScale, cropSize, isDebug=False)
            #
            if rndIsFlip[ii]:
                timgCrop = np.fliplr(timgCrop)
            #
            tmsk0 = (timgCrop[:,:,3].reshape(-1)==128).astype(np.uint8)
            tmsk0[tmsk0>0] = tcls
            tmskCls = np_utils.to_categorical(tmsk0, numCls)
            #
            timg3 = timgCrop[:, :, :3]
            #
            # if isRandomize:
            #     timg3 = preproc_image(timg3)
            timg3 = timg3.astype(np.float32) / 127.5 - 1.0
            if isRandomize:
                timg3 -= rndShiftMean[ii]
                timg3 *= rndShiftStd[ii]
            #
            dataX[ii] = timg3
            dataY[ii] = tmskCls
            if isDebug:
                plt.subplot(2, 2, 1)
                plt.imshow(timg4[:,:,:3])
                plt.gcf().gca().add_artist(plt.Circle(Pxy, Rmin, edgecolor='r', fill=False))
                plt.gcf().gca().add_artist(plt.Circle(Pxy, Rrnd, edgecolor='y', fill=False))
                plt.gcf().gca().add_artist(plt.Circle(Pxy, Rmax, edgecolor='g', fill=False))
                plt.subplot(2, 2, 2)
                plt.imshow(timg4[:,:,3])
                plt.gcf().gca().add_artist(plt.Circle(Pxy, Rmin, edgecolor='r', fill=False))
                plt.gcf().gca().add_artist(plt.Circle(Pxy, Rrnd, edgecolor='y', fill=False))
                plt.gcf().gca().add_artist(plt.Circle(Pxy, Rmax, edgecolor='g', fill=False))
                plt.subplot(2, 2, 3)
                plt.imshow(timgCrop[:, :,:3])
                plt.subplot(2, 2, 4)
                plt.imshow(timgCrop[:, :, 3])
                plt.show()
        yield (dataX, dataY)

#####################################################
if __name__ == '__main__':
    imgSize = 256
    numCls = 4
    # (1) Setup Tran/Validation data
    fidxTrn = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-original-1024x1024-bordered/idx.txt-train.txt'
    fidxVal = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-original-1024x1024-bordered/idx.txt-val.txt'
    wdir = os.path.dirname(fidxTrn)
    # q = next(train_generator(trnX, trnCls, trnInfo, isDebug=False))
    #
    # (2) Input/Output models
    pathModelPrefix = '{0}/model_fcncls_channel'.format(wdir)
    pathModelValLoss = '{0}_valLoss_v1.h5'.format(pathModelPrefix)
    pathModelValAcc = '{0}_valAcc_v1.h5'.format(pathModelPrefix)
    pathModelLatest = '{0}_Latest_v1.h5'.format(pathModelPrefix)
    pathLog = '%s-log.csv' % pathModelValLoss
    # (3) Visualise model (for test)
    #
    # (4) Continue training from checkpoint Model (if exists)
    pathModelRestart = pathModelValLoss
    if not os.path.isfile(pathModelRestart):
        model = buildModelFCNNCLS_UpSampling2D_V3(inpShape=(imgSize, imgSize, 3),
                                                  numCls=numCls)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        pref = time.strftime('%Y.%m.%d-%H.%M.%S')
        pathModelValBk = '%s-%s.bk' % (pathModelValLoss, pref)
        pathModelValAccBk = '%s-%s.bk' % (pathModelValAcc, pref)
        pathModelLatestBk = '%s-%s.bk' % (pathModelLatest, pref)
        shutil.copy(pathModelValLoss, pathModelValBk)
        shutil.copy(pathModelValAcc, pathModelValAccBk)
        # shutil.copy(pathModelLatest, pathModelLatestBk)
        model = keras.models.load_model(pathModelRestart)
        #
        model.compile(optimizer=kopt.Adam(lr=0.00004),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    # (5) Preload data
    trnX, trnCls, _ = readDataImagesCls(fidxTrn, maxNum=None)
    valX, valCls, _ = readDataImagesCls(fidxVal, maxNum=None)
    trnInfo = buildImgInfoList(trnX)
    valInfo = buildImgInfoList(valX)
    numTrn = trnX.shape[0]
    numVal = valX.shape[0]
    #
    batchSize = 32
    numEpochs = 1000
    numIterPerEpoch = numTrn / batchSize
    if numIterPerEpoch<1:
        numIterPerEpoch = 1
    # dataTrn = dataVal
    valXg, valYg = next(train_generator(dataImg=valX,
                                        dataCls=valCls,
                                        dataInfo=valInfo,
                                        imsize=imgSize,
                                        batchSize=1024,
                                        isRandomize=False))
    #
    model.fit_generator(
        generator=train_generator(
                    dataImg=trnX, dataCls=trnCls, dataInfo=trnInfo,
                    imsize=imgSize, batchSize=batchSize, isRandomize=True),
        steps_per_epoch=numIterPerEpoch,
        epochs=numEpochs,
        validation_data=(valXg, valYg),
        callbacks=[
            kall.ModelCheckpoint(pathModelValLoss, verbose=True, save_best_only=True, monitor='val_loss'),
            kall.ModelCheckpoint(pathModelValAcc, verbose=True, save_best_only=True, monitor='val_acc'),
            # kall.ModelCheckpoint(pathModelLatest, verbose=True, save_best_only=False),
            kall.CSVLogger(pathLog, append=True)
        ])
