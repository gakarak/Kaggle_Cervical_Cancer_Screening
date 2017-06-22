#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

runpy='run04_CNN_Cls_Cervix_And_Channel_Inference_MultiFolded_v1.py'

pathModelJson='../models/models-cls-nover-add-5folds.json'
pathTestIdx='test_stage2_original-512x512-acc-bordered/idx.txt'

python ${runpy} ${pathModelJson} ${pathTestIdx} loss

