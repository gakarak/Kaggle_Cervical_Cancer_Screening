#!usr/bin/python
import os
import shutil

import numpy as np

import descriptors as descrs

class DescriptorManager(object):

  def __init__(self, storage_dir):
    self.storage_dir = storage_dir
    self.descriptors = {'LBP' : descrs.buildLbp,
                        'RGB-hist' : descrs.buildRGBHist,
                        'KMeans': descrs.buildKMeansRGBSpaceHist}
    
    if not os.path.exists(self.storage_dir):
      os.makedirs(self.storage_dir)

  def availableDescrs():
    return self.descriptors.keys()

  def load(self, image_name, config):
    filename = self.buildName(image_name, config)
    return np.loadtxt(filename, delimiter=',', dtype=np.float)

  def save(self, image_name, config, vector):
    filename = self.buildName(image_name, config)
    np.savetxt(filename, vector, delimiter=',')

  def paramsToStr(self, params):
    str_params = []
    for key in sorted(params.keys()):

      if key=="mask":
        if (params[key] is not None):
          str_params.append("mask")
      elif "model" in key:
        pass
      else:
        str_params.append(key+"-"+str(params[key]))

    return '_'.join(str_params)

  def buildName(self, image_name, config):
    descr_name = config[0]
    params_str = self.paramsToStr(config[1])
    info = config[2] if len(config) == 3 else ''
    final_name = None
    if info != '':
      final_name = '%s_%s_%s_%s.csv' % (image_name, info, descr_name, params_str)
    else:
      final_name = '%s_%s_%s.csv' % (image_name, descr_name, params_str)
    
    path = os.path.join(self.storage_dir, final_name)
    return path

  def buildFeatures(self, image_name, configs):
    vector = []
    
    for config in configs:
      add_vec = self.load(image_name, config)
      vector = np.concatenate([vector, add_vec])

    return vector

  def calculateFeatures(self, image, config):
    descr_name, params = config[:2]

    describe = self.descriptors[descr_name]

    return describe(image, **params)