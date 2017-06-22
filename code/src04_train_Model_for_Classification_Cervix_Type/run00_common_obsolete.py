#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import multiprocessing as mp
import time
import numpy as np
from run01_CNN_Cls_Cervix_Only_train_v2 import BatchGenerator
from run03_CNN_Cls_Cervix_And_Channel_train_v1 import BatchGeneratorCervixOnly
import threading

class ThreadedDataGeneratorV2(object):
    def __init__(self, nproc=8, isThreadManager=True):
        self._nproc = nproc
        if isThreadManager:
            self._pool = mp.pool.ThreadPool(processes=self._nproc)
        else:
            self._pool = mp.Pool(processes=self._nproc)
        self._batchGenerator = None
        self._cleanData()
        self._genCounter = 0
    def _cleanData(self):
        self._poolStateMerge  = None
        self._poolResultMerge = None
        # gc.collect()
    def isIdle(self):
        if (self._poolStateMerge is not None) and self._poolStateMerge.isAlive():
            return False
        return True
    def setDataGenerator(self, batchGenerator):
        self._batchGenerator = batchGenerator
    def setDataGenerator_OLD(self, dataImg, dataCls, dataImgInfo, imsize = 256,
                               isRandomize=True,
                               angleRange=(-16.,+16.),
                               scaleRange=(0.9, 1.3), fun_random_val=None):
        self._batchGenerator = BatchGeneratorCervixOnly(
                    dataImg=dataImg,
                    dataCls=dataCls,
                    dataImgInfo=dataImgInfo,
                    imsize=imsize,
                    isRandomize=isRandomize,
                    angleRange=angleRange,
                    scaleRange=scaleRange,
                    fun_random_val=fun_random_val)
    def _runner_batch(self, pdata):
        bidx = pdata[0]
        print (':: Batching #{0}'.format(bidx))
        bsiz = pdata[1]
        # dataXY = self._batchGenerator.build_batch(bsiz)
        return self._batchGenerator.build_batch(bsiz)
    def _runner_merge(self, pdata):
        # print ('--------- START MERGE ----------')
        t0 = time.time()
        batchSize = pdata[0]
        tpool = mp.pool.ThreadPool(processes=self._nproc)
        list_batches = tpool.map(self._runner_batch, [(xx, batchSize) for xx in range(self._nproc)])
        numData = len(list_batches[0])
        self._poolResultMerge = [None] * numData
        for ii in range(numData):
            self._poolResultMerge[ii] = np.concatenate([xx[ii] for xx in list_batches])
        # Freeing memory...
        tpool.close()
        dt = time.time() - t0
        # print ('--------- FINISH MERGE ----------')
        self._genCounter += 1
        print ('\tBatched data #{0} is generated: {1:0.3f} (s)'.format(self._genCounter, dt))
    def startBatchGeneration(self, batchSize = 1024):
        bsiz = batchSize/self._nproc
        if self.isIdle():
            self._cleanData()
            self._poolStateMerge = threading.Thread(target=self._runner_merge, args=[(bsiz,)])
            self._poolStateMerge.start()
        else:
            print ('** WARNIG Task Pool is runnig, canceling batchGeneration...')
    def getGeneratedData(self, isClean=True):
        if not self.isIdle():
            return None
        else:
            dataXY = self._poolResultMerge
            if isClean:
                self._cleanData()
            return dataXY
    def toString(self):
        return '::ThreadedDataGenerator isIdle: [{0}], generator=[{1}], #generation = {2}'\
                .format(self.isIdle(), self._batchGenerator, self._genCounter)
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def waitAll(self, dt = 0):
        if not self.isIdle():
            if dt>0:
                time.sleep(dt)
            self._poolStateMerge.join()

class ThreadedDataGenerator():
    def __init__(self, nproc=8, isThreadManager=True):
        self._nproc = nproc
        if isThreadManager:
            self._pool = mp.pool.ThreadPool(processes=self._nproc)
        else:
            self._pool = mp.Pool(processes=self._nproc)
        self._cleanData()
        self._batchGenerator = None
    def _cleanData(self):
        self._poolStates = dict()
        self._poolResultX = dict()
        self._poolResultY = dict()
        for iidx in range(self._nproc):
            self._poolResultX[iidx] = None
            self._poolResultY[iidx] = None
    def setDataGenerator(self, dataImg, dataCls, dataImgInfo, imsize = 256,
                               isRandomize=True,
                               angleRange=(-16.,+16.),
                               scaleRange=(0.9, 1.3), fun_random_val=None):
        self._batchGenerator = BatchGenerator(
                    dataImg=dataImg,
                    dataCls=dataCls,
                    dataImgInfo=dataImgInfo,
                    imsize=imsize,
                    isRandomize=isRandomize,
                    angleRange=angleRange,
                    scaleRange=scaleRange,
                    fun_random_val=fun_random_val)
    def isIdle(self):
        if len(self._poolStates)<1:
            return True
        for kk,vv in self._poolStates.items():
            if not vv.ready():
                return False
        return True
    def _runner(self, pdata):
        print ('******* FFFFUUUCK ********')
        bidx = pdata[0]
        bsiz = pdata[1]
        dataX, dataY = self._batchGenerator.build_batch(bsiz)
        self._poolResultX[bidx] = dataX
        self._poolResultY[bidx] = dataY
    def startBatchGeneration(self, batchSize = 1024):
        bsiz = batchSize/self._nproc
        if self.isIdle():
            self._cleanData()
            for iidx in range(self._nproc):
                self._poolStates[iidx] = self._pool.apply_async(self._runner, [(iidx, bsiz)])
        else:
            print ('** WARNIG Task Pool is runnig, canceling batchGeneration...')
    def getGeneratedData(self):
        if not self.isIdle():
            return None
        else:
            dataX = np.concatenate(self._poolResultX.values())
            dataY = np.concatenate(self._poolResultY.values())
            return (dataX, dataY)
    def toString(self):
        return '::ThreadedDataGenerator isIdle: [{0}], generator=[{1}]'.format(self.isIdle(), self._batchGenerator)
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def waitAll(self, dt = 0):
        if dt>0:
            time.sleep(dt)
            self._pool.close()
            self._pool.terminate()
        else:
            self._pool.close()
            self._pool.join()

if __name__ == '__main__':
    pass