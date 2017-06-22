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
    Flatten, Activation, Reshape, MaxPooling2D, Input, Dense, merge, Dropout, SpatialDropout2D, BatchNormalization
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
        if cc==0:
            tfsiz = (5,5)
        else:
            tfsiz = fsiz
        for ii in range(numConv):
            x = Conv2D(filters=numFlt * (2 **cc), kernel_size=tfsiz,
                       activation='relu',
                       padding=ppadding,
                       W_regularizer=keras.regularizers.l2(0.01))(x)
            # x = BatchNormalization()(x)
            # x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=psiz, padding=ppadding)(x)
    # (2) flatening
    x = Flatten()(x)
    x = Dropout(rate=0.2)(x)
    # (3) hidden dense-layers
    if numHidden is not None:
        if isinstance(numHidden, list):
            for numUnits in numHidden:
                x = Dense(units=numUnits, activation='relu', W_regularizer=keras.regularizers.l2(0.01))(x)

        else:
            x = Dense(units=numHidden, activation='relu',
                      # W_regularizer=keras.regularizers.l2(0.02)
                      )(x)
        x = Dropout(rate=0.5)(x)
    # (4) multiclass-output
    x = Dense(units=numCls, activation='softmax')(x)
    retModel = Model(inputs=dataInput, outputs=x)
    return retModel

#####################################################
def preproc_image(pimg, prnd=None):
    ndim = pimg.ndim
    if prnd is None:
        trnd = np.random.randint(2)
    else:
        trnd = prnd
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
    imgRet = cv2.warpAffine(pimg, matTotal[:2, :], pcropSize)
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
        'r_chn_good': pRelChnSize * r_channel,
        'cnt_chn': rc_channel_cnt,
        'rc_chn': r_channel_good.copy()
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

def buildImgInfoList(dataImg):
    numImg = dataImg.shape[0]
    print (":: Prepare image info ({0})".format(dataImg.shape))
    ret = []
    for ii in range(numImg):
        timg = dataImg[ii]
        tinfo = prepareCervixAndChannelInfo(timg)
        ret.append(tinfo)
        if (ii%10)==0:
            print ('[{0}/{1}]'.format(ii, numImg))
    return ret

#####################################################
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
    dataPaths = lstpath
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
    return dataX, dataY, dataPaths

#####################################################
def getRandomInRange(vrange, pnum=None):
    vmin,vmax = vrange
    if pnum is None:
        trnd = np.random.rand()
    else:
        trnd = np.random.rand(pnum)
    ret = vmin + (vmax-vmin)*trnd
    return ret

def preprocImgForInference(pimg, pinfo, angleRange=(-16.,+16.), batchSize = 16, imsize=256, isRandomize=False):
    sizeCrop = (imsize, imsize)
    dataX = np.zeros((batchSize, imsize, imsize, 3))
    timg = pimg[:, :, :3]
    CNT_chn_rc = pinfo['cnt_chn']
    PTS_chn_rc = pinfo['rc_chn']
    R_chn = pinfo['r_chn_good']
    R_crv = pinfo['r_crv']
    for ii in range(batchSize):
        # R_crop = R_crv
        if R_chn < 10:
            R_chn = 10.
        if isRandomize:
            R_crop = getRandomInRange([0.6 * R_crv, 1.2 * R_crv])
        else:
            R_crop = R_crv
        if PTS_chn_rc.shape[0]>0:
            rndChnPos = np.random.randint(PTS_chn_rc.shape[0])
            P_Center_XY = PTS_chn_rc[rndChnPos][::-1]
        else:
            P_Center_XY = CNT_chn_rc
        #
        if isRandomize:
            angleCrop = getRandomInRange(angleRange)
        else:
            angleCrop = 0.
        scaleCrop2 = (float(imsize) / (2. * R_crop + 2.))
        #
        timgCrop = buildImageWithRotScaleAroundCenter(timg, P_Center_XY, angleCrop, scaleCrop2, sizeCrop, isDebug=False)
        timgCrop = (timgCrop.astype(np.float) / 127.5 - 1.0)
        dataX[ii] = timgCrop
    return dataX

#####################################################
def train_generator_CHANNEL_CLS(dataImg, dataCls, dataImgInfo, batchSize=64, imsize = 256,
                                isRandomize=True,
                                angleRange=(-16.,+16.),
                                scaleRange=(1.0, 1.0), fun_random_val=None):
    numImg   = dataImg.shape[0]
    sizeCrop = (imsize, imsize)
    imgIdx = list(range(numImg))
    while True:
        # rndIdx = np.random.permutation(imgIdx)[:batchSize]
        rndIdx = np.random.randint(0,numImg, batchSize)
        dataX = np.zeros((batchSize, imsize, imsize, 3))
        dataY = np.zeros((batchSize, dataCls.shape[-1]))
        # dataImgG = dataImg[rndIdx]
        rndShiftMean = 0.2*getRandomInRange((-1., 1.0), pnum=batchSize)
        rndShiftStd  = 1.0 + 0.2 * getRandomInRange((-1.0, 1.0), pnum=batchSize)
        #
        for ii, idx in enumerate(rndIdx):
            # timg = dataImgG[ii][:,:,:3]
            timg = dataImg[idx][:, :, :3]
            tinf = dataImgInfo[idx]
            CNT_chn_rc = tinf['cnt_chn']
            PTS_chn_rc = tinf['rc_chn']
            R_chn = tinf['r_chn_good']
            R_crv = tinf['r_crv']
            # R_crop = R_crv
            if R_chn<10:
                R_chn = 10.
            R_crop = getRandomInRange([0.6*R_crv, 1.2*R_crv])
            # R_crop = R_chn * 3.
            # ----
            # if R_chn<10:
            #     R_chn = 10.
            # K_chn2crv = float(R_chn)/float(R_crv)
            # K_max = 3.
            # if K_chn2crv>K_max:
            #     R_crop = R_chn * K_max
            # ----
            rndChnPos = np.random.randint(PTS_chn_rc.shape[0])
            P_Center_XY = PTS_chn_rc[rndChnPos][::-1]
            #
            if isRandomize:
                angleCrop = getRandomInRange(angleRange)
                scaleCrop = getRandomInRange(scaleRange)
            else:
                angleCrop = 0.
                scaleCrop = 1.
            scaleCrop2 = scaleCrop * (float(imsize)/(2.*R_crop + 2.))
            #
            timgCrop = buildImageWithRotScaleAroundCenter(timg, P_Center_XY, angleCrop, scaleCrop2, sizeCrop, isDebug=False)
            if fun_random_val is not None:
                timgCrop = fun_random_val(timgCrop)
            timgCrop = (timgCrop.astype(np.float)/127.5 - 1.0)
            if isRandomize:
                timgCrop -= rndShiftMean[ii]
                timgCrop *= rndShiftStd[ii]
            dataX[ii] = timgCrop
            dataY[ii] = dataCls[idx]
        yield (dataX, dataY)

#####################################################
if __name__ == '__main__':
    # (1) Setup Tran/Validation data
    fidxTrn = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/01-data-512x512/idx.txt-train.txt'
    fidxVal = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-processed-stage2/01-data-512x512/idx.txt-val.txt'
    wdir = os.path.dirname(fidxTrn)
    # (2) Input/Output models
    pathModelValLoss = '{0}/model_CNN_Classification_valLoss_v1.h5'.format(wdir)
    pathModelValAcc = '{0}/model_CNN_Classification_valAcc_v1.h5'.format(wdir)
    pathModelLatest = '{0}/model_CNN_Classification_Latest_v1.h5'.format(wdir)
    pathLog = '%s-log.csv' % pathModelValLoss
    # (3) Continue training from checkpoint Model (if exists)
    pathModelRestart = pathModelValLoss
    if not os.path.isfile(pathModelRestart):
        print (':: Trained model not found: build new model...')
        model = buildModelCNN_Classification(numConv=1, ppadding='same', numHidden=128)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        print ('!!! WARNING !!! Found trained model, loading... [{0}]'.format(pathModelRestart))
        pref = time.strftime('%Y.%m.%d-%H.%M.%S')
        pathModelValBk = '%s-%s.bk' % (pathModelValLoss, pref)
        pathModelValAccBk = '%s-%s.bk' % (pathModelValAcc, pref)
        pathModelLatestBk = '%s-%s.bk' % (pathModelLatest, pref)
        shutil.copy(pathModelValLoss, pathModelValBk)
        shutil.copy(pathModelValAcc, pathModelValAccBk)
        # shutil.copy(pathModelLatest, pathModelLatestBk)
        model = keras.models.load_model(pathModelRestart)
    model.summary()
    # (4) Preload data
    trnX, trnY, _ = readDataImagesCls(fidxTrn)
    valX, valY, _ = readDataImagesCls(fidxVal) #, maxNum=10)
    # trnX, trnY = valX, valY
    trnInfo = buildImgInfoList(trnX)
    valInfo = buildImgInfoList(valX)
    # (5) prepare image generator
    numTrn = trnX.shape[0]
    numVal = valX.shape[0]
    imgSize = 256
    batchSize = 64
    numEpochs = 1000
    numIterPerEpochTrn = 2 * numTrn / batchSize
    numIterPerEpochVal = 1 * numVal / batchSize
    if numIterPerEpochTrn<1:
        numIterPerEpochTrn = 1
    generatorTrn = train_generator_CHANNEL_CLS(dataImg=trnX, dataCls=trnY, dataImgInfo=trnInfo,
                                                 batchSize=batchSize,
                                                 isRandomize=True,
                                                 fun_random_val=None,
                                                 # fun_random_val=preproc_image
                                               )
    generatorVal = train_generator_CHANNEL_CLS(dataImg=valX, dataCls=valY, dataImgInfo=valInfo,
                                               batchSize=1024,
                                                # batchSize=batchSize,
                                                isRandomize=False, fun_random_val=None)
    # (6) Generate fixed validation data
    valX_ext, valY_ext = next(generatorVal)
    # (7) Train model
    model.fit_generator(
        generator=generatorTrn,
        steps_per_epoch=numIterPerEpochTrn,
        epochs=numEpochs,
        validation_data=(valX_ext, valY_ext),
        # validation_data=generatorVal,
        # validation_steps=numIterPerEpochVal,
        callbacks=[
            kall.ModelCheckpoint(pathModelValLoss, verbose=True, save_best_only=True, monitor='val_loss'),
            kall.ModelCheckpoint(pathModelValAcc,  verbose=True, save_best_only=True, monitor='val_acc'),
            # kall.ModelCheckpoint(pathModelLatest,  verbose=True, save_best_only=False),
            kall.CSVLogger(pathLog, append=True)
        ])
