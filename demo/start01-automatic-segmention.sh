#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

runpy='run01_generate_automasked_testdata_v2_MultiFolded.py'

pathModelJson='../models/models-segm.json'
pathTestIdx='test_stage2_original/idx.txt'

python ${runpy} ${pathModelJson} ${pathTestIdx} 512 acc brd
##python ${runpy} ${pathModelJson} ${pathTestIdx} 512 loss brd
