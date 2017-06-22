# Python 2/3 compatibility
from __future__ import print_function

import sys
import os
import random
import math

# import cv2
import numpy as np
import matplotlib.pyplot as plt

import skimage.io as skio
from skimage import feature
from skimage.color import rgb2gray

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from descriptor_manager import DescriptorManager
import utilities as utils

if (__name__ == '__main__'):
    wdir = '../../dataset/train-x512-processed-stage2'
    train_idx = os.path.join(wdir, 'idx_all.txt-shuf.txt')

    print('Reading data...')
    cervixes = utils.readCervixes(train_idx, "cervix", with_imgs=False)

    configs = [
                ['RGB-hist', {'bins': 32, 'mask': True}],
                # ['RGB-hist', {'bins': 128, 'mask': True}],
                # ['RGB-hist', {'bins': 256, 'mask': True}],
                ['LBP', {'radius': 12, 'numPoints': 36, 'mask': True}],
                # ['LBP', {'radius': 16, 'numPoints': 48, 'mask': True}],
                # ['LBP', {'radius': 32, 'numPoints': 160, 'mask': True}],
                ['RGB-hist', {'bins': 32, 'mask': True}, "channel-mask"],
                ['RGB-hist', {'bins': 128, 'mask': True}, "channel-mask"],
                ['RGB-hist', {'bins': 256, 'mask': True}, "channel-mask"],
                ['LBP', {'radius': 12, 'numPoints': 36, 'mask': True}, "channel-mask"],
                ['LBP', {'radius': 16, 'numPoints': 48, 'mask': True}, "channel-mask"],
                ['LBP', {'radius': 32, 'numPoints': 160, 'mask': True}, "channel-mask"],
                # ['KMeans', {'kmeans_model': True, 'mask': True}, "cervix-20000-512"],
                # ['KMeans', {'kmeans_model': True, 'mask': True}, "cervix-20000-256"],
                ['KMeans', {'kmeans_model': True, 'mask': True}, "cervix-20000-64"],
                # ['KMeans', {'kmeans_model': True, 'mask': True}, "cervix-20000-16"],
                # ['KMeans', {'kmeans_model': True, 'mask': True}, "cervix-50000-64"],
                ['KMeans', {'kmeans_model': True, 'mask': True}, "channel-20000-64"],
            ]

    print('Loading descriptors...')
    manager = DescriptorManager(os.path.join(wdir, 'descriptors'))

    cervixes_data = []
    cervixes_labels = []
    for c in cervixes:
        cervixes_data.append(manager.buildFeatures(c['code_name'], configs))
        cervixes_labels.append(c['type'])

    print('Feature length: ' + str(cervixes_data[0].shape))

    FOLD_SIZE = 5
    BATCH_SIZE = len(cervixes_data) // FOLD_SIZE
    WIN_PROB = 0.66
    LOS_PROB = (1 - WIN_PROB) / 2

    print('Start %d-fold cross-validation' % FOLD_SIZE)
    print('Manual win probability: %f' % WIN_PROB)
    #random.shuffle(cervixes)
    average_acc = 0
    average_log_loss_cls = 0
    average_log_loss_man = 0
    for fold_i in range(FOLD_SIZE):
        print('[%d / %d] %s' % (fold_i+1, FOLD_SIZE, 'folds'))
        valid_start_i = fold_i * BATCH_SIZE
        valid_finish_i = (fold_i + 1) * BATCH_SIZE
        
        if fold_i == FOLD_SIZE - 1:
            valid_finish_i = len(cervixes)
        
        train_data = cervixes_data[:valid_start_i] + cervixes_data[valid_finish_i:]
        train_labels = cervixes_labels[:valid_start_i] + cervixes_labels[valid_finish_i:]

        valid_data = cervixes_data[valid_start_i:valid_finish_i]
        valid_labels = cervixes_labels[valid_start_i:valid_finish_i]

        #model = svm.LinearSVC(C=100.0, random_state=42)
        model = RandomForestClassifier(max_depth = 35, n_estimators=1000, max_features = 64, random_state=42)
        # model = AdaBoostClassifier(learning_rate=0.1, n_estimators=1000)
        model.fit(train_data, train_labels)

        # loop over the testing images
        is_log_preds = True

        try:
            predictions = model.predict_log_proba(valid_data)
        except:
            is_log_preds = False
            predictions = model.predict(valid_data)

        accuracy = 0.
        log_loss_cls = 0.
        log_loss_man = 0.

        for idx, [real_type, log_pred] in enumerate(zip(valid_labels, predictions)):
            pred_type = log_pred
            if (is_log_preds):
                log_loss_cls += -log_pred[real_type - 1]
                pred_type = np.argmax(log_pred) + 1


            if real_type == pred_type:
                accuracy += 1
                log_loss_man += -math.log(WIN_PROB)
            else:
                log_loss_man += -math.log(LOS_PROB)

        accuracy /= len(valid_labels)
        log_loss_cls /= len(valid_labels)
        log_loss_man /= len(valid_labels)
        
        print('\tAccuracy: %f' % accuracy)
        print('\tManual log loss: %f' % log_loss_man)
        if is_log_preds:
            print('\tClassificator log loss: %f' % log_loss_cls)

        average_acc += accuracy / FOLD_SIZE
        average_log_loss_man += log_loss_man / FOLD_SIZE
        average_log_loss_cls += log_loss_cls / FOLD_SIZE

    print('Average acc.: %f' % average_acc)
    print('Average manual (win_prob=%f) log loss: %f' % (WIN_PROB, average_log_loss_man))
    print('Average clasificator log loss: %f' % average_log_loss_cls)

    # plt.figure()

    # class_names = ["Type_1", "Type_2", "Type_3"]
    # cnf_matrix = confusion_matrix(valid_labels, predictions)

    # utils.plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='CNN, Accuracy = %f' % accuracy)

    # plt.show()