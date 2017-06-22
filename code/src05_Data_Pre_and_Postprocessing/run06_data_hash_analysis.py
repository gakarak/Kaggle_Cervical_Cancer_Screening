#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fidxAdd='/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_stage1/idxmd5_data_stage1.txt'
    fidxStage2='/home/ar/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data_stage2/test_stg2/test_stage2_original/idxmd5_test_stage2_original.txt'
    #
    dataAdd=pd.read_csv(fidxAdd)
    dataStage2=pd.read_csv(fidxStage2)
    #
    hashAdd = dataAdd['hash'].as_matrix()
    hashStage2 = dataStage2['hash'].as_matrix()
    numAdd = len(hashAdd)
    numStage2 = len(hashStage2)
    matCmp = np.zeros((numStage2, numAdd))
    for iadd in range(numAdd):
        for iStage2 in range(numStage2):
            if hashAdd[iadd]==hashStage2[iStage2]:
                matCmp[iadd, iStage2] = 1
        print ('\t[{0}/{1}]'.format(iadd, numAdd))
    plt.imshow(matCmp)
    plt.show()
    print ('-')
