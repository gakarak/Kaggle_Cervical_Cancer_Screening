#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#########################################
def split_list_by_blocks(lst, psiz):
    tret = [lst[x:x + psiz] for x in range(0, len(lst), psiz)]
    return tret

#########################################
if __name__ == '__main__':
    fidx = '/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_additional/01_train_add-x512-original-bordered_Results/idx.txt'
    numFolds = 5
    if len(sys.argv)>1:
        fidx = sys.argv[1]
    if len(sys.argv)>2:
        numFolds = int(sys.argv[2])
    if not os.path.isfile(fidx):
        raise Exception('*** Cant find index file! [{0}]'.format(fidx))
    #
    print (':: Index file: [{0}]\n\t#Folds = {1}'.format(fidx, numFolds))
    #
    data = pd.read_csv(fidx)
    numData = len(data)
    idxData = list(range(numData))
    idxData = np.random.permutation(idxData).tolist()
    sizSplit = int(math.ceil(float(numData)/numFolds))
    splitIdxVal = split_list_by_blocks(idxData, sizSplit)
    for isplit, splitIdxVal in enumerate(splitIdxVal):
        splitIdxTrn = list(set(range(numData)) - set(splitIdxVal))
        fidxTrn = '{0}_fold{1}_trn.csv'.format(fidx, isplit)
        fidxVal = '{0}_fold{1}_val.csv'.format(fidx, isplit)
        dataTrn = data.irow(splitIdxTrn)
        dataVal = data.irow(splitIdxVal)
        dataTrn.to_csv(fidxTrn, index=False)
        dataVal.to_csv(fidxVal, index=False)
        print ('\t[{0}/{1}]'.format(isplit, numFolds))