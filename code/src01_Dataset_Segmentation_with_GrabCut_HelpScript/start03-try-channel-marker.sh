#!/bin/bash

wdir='/home/ar/data/@Kaggle/01_Intel_&_MobileODT_Cervical_Cancer_Screening/data/train-x512-stage2/Type_3'

##python run01_segmentator_grabcut_v1.py ../data/test01_Train_Samples/Type_1/
python channel_marker.py "${wdir}"
