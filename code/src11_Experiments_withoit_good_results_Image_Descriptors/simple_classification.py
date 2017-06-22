import sys
import os

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
    valid_idx = os.path.join(wdir, 'idx_test.txt')
    # valid_idx = 'idx-mask.txt-val.txt-processed.txt'

    print('Reading data...')

    train = utils.readCervixes(train_idx, "cervix", with_imgs=False)
    valid = utils.readCervixes(valid_idx, "test", with_imgs=False)
    # valid = utils.readCervixes('/home/ar/bitbucket.org/kaggle_intel_image_preprocessing-Dataset/train-x512-processed', valid_idx, "CNN")

    manager = DescriptorManager(os.path.join(wdir, 'descriptors'))

    configs = [
                ['RGB-hist', {'bins': 32, 'mask': True}],
                ['RGB-hist', {'bins': 128, 'mask': True}],
                ['RGB-hist', {'bins': 256, 'mask': True}],
                ['LBP', {'radius': 12, 'numPoints': 36, 'mask': True}],
                ['LBP', {'radius': 16, 'numPoints': 48, 'mask': True}],
                ['LBP', {'radius': 32, 'numPoints': 160, 'mask': True}],
                # ['RGB-hist', {'bins': 32, 'mask': True}, "channel-mask"],
                # ['RGB-hist', {'bins': 128, 'mask': True}, "channel-mask"],
                # ['RGB-hist', {'bins': 256, 'mask': True}, "channel-mask"],
                # ['LBP', {'radius': 12, 'numPoints': 36, 'mask': True}, "channel-mask"],
                # ['LBP', {'radius': 16, 'numPoints': 48, 'mask': True}, "channel-mask"],
                # ['LBP', {'radius': 32, 'numPoints': 160, 'mask': True}, "channel-mask"],
                # ['KMeans', {'kmeans_model': True, 'mask': True}, "cervix-20000-512"],
                # ['KMeans', {'kmeans_model': True, 'mask': True}, "cervix-20000-256"],
                ['KMeans', {'kmeans_model': True, 'mask': True}, "cervix-20000-64"],
                # ['KMeans', {'kmeans_model': True, 'mask': True}, "cervix-20000-16"],
                # ['KMeans', {'kmeans_model': True, 'mask': True}, "cervix-50000-64"],
                # ['KMeans', {'kmeans_model': True, 'mask': True}, "channel-20000-64"],
            ]

    print('Loading descriptors...')

    train_data = []
    train_labels = []
    for t in train:
        train_data.append(manager.buildFeatures(t['code_name'], configs))
        train_labels.append(t['type'])

    valid_data = []
    valid_labels = []
    for v in valid:
        valid_data.append(manager.buildFeatures(v['code_name'], configs))
        valid_labels.append(v['type'])

    print('Data loaded, feature length: ' + str(train_data[0].shape))

    #model = svm.LinearSVC(C=100.0, random_state=42)
    #model = RandomForestClassifier(max_depth = 15, n_estimators=500, max_features = 100, random_state=42)
    model = RandomForestClassifier(max_depth = 15, n_estimators=500, max_features = 128, random_state=42)
    # model = AdaBoostClassifier(learning_rate=0.1, n_estimators=1000)
    model.fit(train_data, train_labels)


    # loop over the testing images
    # predictions = model.predict(valid_data)
    predictions = model.predict_proba(valid_data)

    w_prob = 0.63
    l_prob = (1-w_prob)/2

    accuracy = 0
    with open('result-forest-only-cervix.csv', 'w') as f:#% w_prob, 'w') as f:
        f.write('image_name,Type_1,Type_2,Type_3\n')
        for idx, [v, p] in enumerate(zip(valid_labels, predictions)):
            arr = [valid[idx]['image_name'], str(p[0]), str(p[1]), str(p[2])]
            # arr = [valid[idx]['image_name'], str(l_prob), str(l_prob), str(l_prob)]
            # arr[p] = str(w_prob)
            f.write(','.join(arr))
            f.write('\n')

            # if (v == p):
            #     accuracy += 1
    accuracy /= len(valid_labels)

    # plt.figure()

    # class_names = ["Type_1", "Type_2", "Type_3"]
    # cnf_matrix = confusion_matrix(valid_labels, predictions)

    # utils.plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='CNN, Accuracy = %f' % accuracy)

    # plt.show()