import os
import itertools

import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt

CERVIX_ALPHA = 255
CHANNEL_ALPHA = 128
BACKGROUND_ALPHA = 64

def createName(path_to_img, type_cls, dataset_name):
    img_name = os.path.basename(path_to_img)
    
    name = "type"+str(type_cls)+"_"+img_name
    if (dataset_name is not None and dataset_name != ''):
        name = dataset_name + "_" + name

    return name

def readCervixes(path_to_idx, dataset_name='', with_imgs=True):
    idx_folder = os.path.split(path_to_idx)[0]

    cervixes = []
    with open(path_to_idx) as idx:
        next(idx) #read header
        for line in idx:
            [fimg, ftype] = line.split(',')
            full_path = os.path.join(idx_folder, fimg)

            img = None
            cervix_mask = None
            channel_mask = None

            if  with_imgs:
              img = np.array(skio.imread(full_path))
              has_mask = img.shape[2] == 4
              if has_mask:
                  cervix_mask = img[:,:,3] == CERVIX_ALPHA
                  channel_mask = img[:,:,3] == CHANNEL_ALPHA
                  img = img[:,:,0:3]

            cls_type = int(ftype)
            
            cervixes.append({'image': img, 
                             'cervix_mask': cervix_mask,
                             'channel_mask': channel_mask,
                             'type': cls_type,
                             'full_path': full_path, 
                             'dataset_name': dataset_name, 
                             'code_name': createName(fimg, cls_type, dataset_name),
                             'image_name': os.path.basename(fimg).split('.')[0] + '.jpg'
                             })

    return cervixes

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')