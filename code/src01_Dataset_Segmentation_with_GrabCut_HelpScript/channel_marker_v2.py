#!/usr/bin/env python
"""
Key '0' - To select areas of background
Key '1' - To select areas of cervix
Key '2' - To select areas of channel

Key 'l' - go to next image
Key 'k' - go to previous iamge

Key 'd' - inc thickness
Key 'a' - dec thickness

Key 'r' - To reset mask
Key 's' - To save the results
Key 'm' - move processed image

Usage:
    channel_marker {/path/to/directory-with-images-in-PNG}
"""

# Python 2/3 compatibility
from __future__ import print_function

import os
import numpy as np
import cv2
import sys
import glob
import shutil
import matplotlib.pyplot as plt


def_win_input_img = 'input-image'
def_win_output_msk = 'output-mask'

def_BLACK = [0, 0, 0]
def_CHANNEL_COLOR = [120, 155, 75]
def_CERVIX_COLOR = [255, 255, 255]

def_DRAW_BG = {'color' : def_BLACK, 'val' : 0, 'alpha': 64}
def_DRAW_CERVIX = {'color' : def_CERVIX_COLOR, 'val' : 1, 'alpha': 255}
def_DRAW_CHANNEL = {'color' : def_CHANNEL_COLOR, 'val' : 2, 'alpha': 128}

def_DRAWS = [def_DRAW_BG, def_DRAW_CHANNEL]
def_ALPHA_TO_VAL = [def_DRAW_BG, def_DRAW_CERVIX, def_DRAW_CHANNEL]

globX = 10
globY = 10
is_first_run = True

#######################
class Dataset:
    _ix = None
    _iy = None
    #
    mskPrefix = '-processed.png'
    outPrefix = 'processed-images'
    denPrefix = 'denied'
    wdir = None
    pathImgs = None
    cidx = None
    #
    _orig = None
    _img_rgb = None
    _msk = None
    _img_draw = None
    _is_drawing = False
    _value = None
    #
    _marker_sizes = [9, 9, 9]
    _marker_idx = 0
    #
    def __init__(self, pdir=None):
        self.setWorkDirectory(pdir=pdir)
    def setWorkDirectory(self, pdir=None):
        if pdir is not None:
            self.wdir = os.path.abspath(pdir)
            self.cidx = 0
            self.updateImages()
    def updateImages(self):
        if (self.wdir is not None):
            tlst = glob.glob('{0}/*.jpg-automasked.png'.format(self.wdir))
            self.pathImgs = np.array(sorted(tlst))
            if (self.cidx > len(tlst)):
                self.cidx = 0
            self._value = def_DRAW_CHANNEL #2
            self._marker_idx = 2
            # update_mouse_size()
    def isOk(self):
        return (self.wdir is not None) and (self.pathImgs is not None)
    def getNumImages(self):
        if self.isOk():
            return len(self.pathImgs)
    def getCurrentIdx(self):
        return self.cidx
    def getNumProcessedImages(self):
        if self.isOk():
            if os.path.isdir(self.outputDir()):
                lstImg = glob.glob('{0}/*.jpg-automasked.png'.format(self.outputDir()))
                return len(lstImg)
            else:
                return 0
    def prevImageIdx(self):
        if self.isOk():
            self.cidx -=1
            if self.cidx<0:
                self.cidx = self.getNumImages() - 1
            return True
        return False
    def nextImageIdx(self):
        if self.isOk():
            self.cidx +=1
            if self.cidx>=self.getNumImages():
                self.cidx = 0
            return True
        return False
    def _getPathImg(self):
        return str(self.pathImgs[self.cidx])
    def _getPathImgProc(self):
        return self._getPathMsk()
    def _getPathMsk(self):
        return '{0}{1}'.format(self._getPathImg(), self.mskPrefix)
    def outputDir(self):
        if self.isOk():
            outDir = os.path.join(self.wdir, self.outPrefix)
            if not os.path.isdir(outDir):
                os.makedirs(outDir)
            return outDir
    def deniedDir(self):
        if self.isOk():
            denDir = os.path.join(self.wdir, self.denPrefix)
            if not os.path.isdir(denDir):
                os.makedirs(denDir)
            return denDir
    def _toString(self):
        if not self.isOk():
            return 'Dataset is not initialized...'
        else:
            return 'Dataset: #Images/#Processed = {0}/{1}, current = {2}'.format(
                self.getNumImages(),
                self.getNumProcessedImages(),
                self.getCurrentIdx())
    def __str__(self):
        return self._toString()
    def __repr__(self):
        return self._toString()
#
    def loadCurretImage(self):
        if self.isOk():
            if self.getNumImages()<1:
                print (' !!! WARNING !!! cant find files in directory [{0}], skip...'.format(self.wdir))
                return

            self._orig = cv2.imread(self._getPathImg(), cv2.IMREAD_UNCHANGED)
            self._img_rgb = self._orig[:,:,:3].copy()
            self.resetMask()

    def resetMask(self):
        tmsk = self._orig[:,:,3]
        # convert binary mask to Grab-Mask
        self._msk = np.zeros(tmsk.shape, np.uint8)
        for mask_type in def_ALPHA_TO_VAL:
            self._msk[tmsk == mask_type['alpha']] = mask_type['val']

        self._img_draw = draw_mask_on_image(self._img_rgb, self._msk)
        self._is_drawing = False

    def saveMasked(self):
        if self.isOk():
            retMsk = np.zeros(self._msk.shape, dtype=np.uint8)
            
            for mask_type in def_ALPHA_TO_VAL:
                retMsk[self._msk == mask_type['val']] = mask_type['alpha']
            
            retMasked = np.dstack( (self._img_rgb, retMsk) )
            fout = self._getPathImgProc()
            cv2.imwrite(fout, retMasked)
            print (':: SAVE to [{0}]'.format(fout))

    def moveProcessedImage(self):
        if self.isOk():
            pathImg = self._getPathImg()
            pathMsk = self._getPathImgProc()
            dirOut = self.outputDir()
            if os.path.isfile(pathImg) and os.path.isfile(pathMsk):
                shutil.move(pathImg, dirOut)
                shutil.move(pathMsk, dirOut)
                self.updateImages()
                self.loadCurretImage()
                print (':: MOVE from [{0}] to [{1}]'.format(pathMsk, dirOut))
            else:
                print ('\t***Image is not processed!, skip... [{0}]'.format(pathMsk))
            print(self._toString())

    def moveDeniedImage(self):
        if self.isOk():
            pathImg = self._getPathImg()
            pathMsk = self._getPathImgProc()
            dirDen = self.deniedDir()
            if os.path.isfile(pathImg):
                shutil.move(pathImg, dirDen)
                if os.path.isfile(pathMsk):
                    shutil.move(pathMsk, dirDen)
                self.updateImages()
                self.loadCurretImage()
                print ('DENIED :: MOVE from [{0}] to [{1}]'.format(pathImg, dirDen))
            else:
                print ('\t***Image is not processed!, skip... [{0}]'.format(pathImg))

def draw_mask_on_image(pimg, pmsk):
    img_with_mask = pimg.copy()

    for draw in def_DRAWS:
        img_with_mask[pmsk == draw['val']] = draw['color']

    return img_with_mask

def mark_position(pdataset, px, py, pthickness):
    cv2.circle(pdataset._img_draw, (px, py), pthickness, pdataset._value['color'], -1)
    cv2.circle(pdataset._msk, (px, py), pthickness, pdataset._value['val'], -1)

def draw_all_windows():
    global globX, globY, is_first_run, datasetPtr
    if is_first_run:
        cv2.namedWindow(def_win_input_img, cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(def_win_output_msk, cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(def_win_input_img, on_mouse)
        cv2.setMouseCallback(def_win_output_msk, on_mouse)
        cv2.moveWindow(def_win_input_img, 600, 50)
        cv2.createTrackbar('Radius', def_win_output_msk, 0, 50, on_track_mouse_size)
        cv2.setTrackbarPos('Radius', def_win_output_msk, thickness)
        is_first_run = False
    if datasetPtr.isOk:
        timg1 = datasetPtr._img_draw.copy()
        #timg1 = datasetPtr._img_rgb.copy()
        timg2 = datasetPtr._img_rgb.copy()
        # (0) prepare masked images
        for mask_type in def_DRAWS:
            c_img = timg2.copy()
            c_img[datasetPtr._msk == mask_type['val']] = np.array(mask_type['color'])
            timg2 = cv2.addWeighted(timg2, 0.6, c_img, 0.4, 0)
        # (1) draw current mouse pointer
        colorCircle = datasetPtr._value['color']
        cv2.circle(timg1, (globX, globY), thickness, colorCircle)
        cv2.circle(timg1, (globX, globY), 1, colorCircle)
        cv2.circle(timg2, (globX, globY), thickness, colorCircle)
        #
        cv2.imshow(def_win_input_img, timg1)
        cv2.imshow(def_win_output_msk, timg2)

def on_mouse(event, x, y, flags, param):
    global globX, globY, datasetPtr
    globX = x
    globY = y
    # draw touchup curves
    if event == cv2.EVENT_LBUTTONDOWN:
        datasetPtr._is_drawing = True
        mark_position(datasetPtr, x,y, thickness)
    elif event == cv2.EVENT_MOUSEMOVE:
        if datasetPtr._is_drawing == True:
            mark_position(datasetPtr, x, y, thickness)
    elif event == cv2.EVENT_LBUTTONUP:
        if datasetPtr._is_drawing == True:
            datasetPtr._is_drawing = False
            mark_position(datasetPtr, x, y, thickness)
    draw_all_windows()

################################
# setting up flags
drawing = False         # flag for drawing curves
thickness = 9           # brush thickness
datasetPtr = Dataset() #None

################################
def on_track_mouse_size(x):
    global thickness, datasetPtr
    thickness = x + 1
    datasetPtr._marker_sizes[datasetPtr._marker_idx] = thickness - 1

def update_mouse_size():
    global thickness, datasetPtr
    thickness = datasetPtr._marker_sizes[datasetPtr._marker_idx]
    cv2.setTrackbarPos('Radius', def_win_output_msk, thickness)

################################
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ('Usage: {0} {{/path/to/dir/with/images}}'.format(sys.argv[0]))
        sys.exit(0)
    else:
        datasetPtr.setWorkDirectory(pdir=sys.argv[1]) # Dataset(pdir=sys.argv[1])

    print (datasetPtr)
    datasetPtr.loadCurretImage()
    draw_all_windows()

    while True:
        k = cv2.waitKey() & 255
        # key bindings
        if k == 27:  # esc to exit
            break
        elif k == ord('0'):
            print(" mark BACKGROUND regions with left mouse button \n")
            datasetPtr._value = def_DRAW_BG
            datasetPtr._marker_idx = 0
            update_mouse_size()
        elif k == ord('1'):
            print(" mark CERVIX regions with left mouse button \n")
            datasetPtr._value = def_DRAW_CERVIX
            datasetPtr._marker_idx = 1
            update_mouse_size()
        elif k == ord('2'):
            print(" mark CHANNEL regions with left mouse button \n")
            datasetPtr._value = def_DRAW_CHANNEL
            datasetPtr._marker_idx = 2
            update_mouse_size()
        elif k == ord('r'):
            datasetPtr.resetMask()
        elif k == ord('k'):
            if datasetPtr.prevImageIdx():
                datasetPtr.loadCurretImage()
            else:
                print ('!!! Cannt load previous image: {0}'.format(datasetPtr._toString()))
            update_mouse_size()
        elif k == ord('l'):
            if datasetPtr.nextImageIdx():
                datasetPtr.loadCurretImage()
            else:
                print ('!!! Cannt load next image: {0}'.format(datasetPtr._toString()))
            update_mouse_size()
        elif k == ord('s'):
            datasetPtr.saveMasked()
            update_mouse_size()
        elif k == ord('m'):
            datasetPtr.moveProcessedImage()
            print(datasetPtr)
            update_mouse_size()
        elif k == ord('d'):
            datasetPtr.moveDeniedImage()
            print(datasetPtr)
        elif k == ord('h'):
            print (__doc__)
        elif k == ord('e'):
            thickness += 1
        elif k == ord('q'):
            thickness -= 1
        

        draw_all_windows()

