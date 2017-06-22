#!/bin/bash

foutIdx="idxmd5_$(basename $PWD).txt"


md5sum ./imgs/*.jpg | tee -a ${foutIdx}
