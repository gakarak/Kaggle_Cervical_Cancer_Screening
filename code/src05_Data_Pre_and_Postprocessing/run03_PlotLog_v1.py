#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
import glob
import skimage.io as skio

#############################
def press_event(event):
    global currentLogIdx
    print('\t--> press-event: {0}'.format(event.key))
    sys.stdout.flush()
    if event.key == 'k':
        currentLogIdx -= 1
        if currentLogIdx < 0:
            currentLogIdx = numLogs - 1
    if event.key == 'l':
        currentLogIdx += 1
        if currentLogIdx >= numLogs:
            currentLogIdx = 0

#############################
if __name__ == '__main__':
    currentLogIdx = 0
    wdir = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/01_train_add-x512-original-bordered_Results/models_bk1'
    if len(sys.argv) > 1:
        wdir = sys.argv[1]
    else:
        print ('*** Usage: {0} [/path/to/dir-with-logs]'.format(os.path.basename(sys.argv[0])))
    if not os.path.isdir(wdir):
        raise Exception('\tCant find directory with logs! [{0}]'.format(wdir))
    #
    logSuffix = '-log.csv'
    listLogs = glob.glob('{0}/*{1}'.format(wdir, logSuffix))
    numLogs = len(listLogs)
    if numLogs<1:
        raise Exception('\tCant find LOG file in format [*{0}] in directory [{1}]'.format(logSuffix, wdir))
    plt.figure()
    ptrFigure = plt.gcf()
    ptrFigure.canvas.mpl_connect('key_press_event', press_event)
    cnt = 0
    #
    while True:
        flog = listLogs[currentLogIdx]
        bname = os.path.basename(flog)
        ptrFigure.canvas.set_window_title('Current: [{0}]'.format(bname))
        if os.path.isfile(flog):
            data = pd.read_csv(flog)
            # dataIter = data['iter'].as_matrix()
            dataLossTrn = data['loss'].as_matrix()
            dataLossVal = data['val_loss'].as_matrix()
            dataAccTrn = data['acc'].as_matrix()
            dataAccVal = data['val_acc'].as_matrix()
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.plot(dataLossTrn)
            plt.plot(dataLossVal)
            plt.grid(True)
            plt.legend(['loss-train', 'loss-validation'], loc='best')
            plt.title('::Loss')
            #
            plt.subplot(1, 2, 2)
            plt.plot(dataAccTrn)
            plt.plot(dataAccVal)
            plt.grid(True)
            plt.legend(['acc-train', 'acc-validation'], loc='best')
            plt.title('::Accuracy')
            #
            plt.show(block=False)
            plt.pause(5)
            print (':: update: [{0}]'.format(cnt))
            cnt += 1
        else:
            print ('*** WARNING *** cant find log-file [{0}]'.format(flog))
