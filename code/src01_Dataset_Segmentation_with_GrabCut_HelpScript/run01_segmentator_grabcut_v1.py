#!/usr/bin/env python
"""
Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'l' - go to next image
Key 'k' - go to previous iamge

Key 'n' - To update the segmentation
Key 'r' - To reset mask
Key 's' - To save the results
Key 'm' - move processed image
Key 'd' - deny image (as bad)

Usage:
    Segmentator_GrabCut {/path/to/directory-with-images-in-JPG}
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
def_win_output_mask_bg = 'output-mask-bg'

def_BBOX_PERCENT = 0.1
def_WORK_DIR = '/Users/alexanderkalinovsky/tmp/000/Type_1'
def_MSK_THRESHOLD = 64
def_BLUE = [255, 0, 0]        # rectangle color
def_RED = [0, 0, 255]         # PR BG
def_GREEN = [0, 255, 0]       # PR FG
def_BLACK = [0, 0, 0]         # sure BG
def_WHITE = [255, 255, 255]   # sure FG

def_DRAW_BG = {'color' : def_BLACK, 'val' : 0}
def_DRAW_FG = {'color' : def_WHITE, 'val' : 1}
def_DRAW_PR_FG = {'color' : def_GREEN, 'val' : 3}
def_DRAW_PR_BG = {'color' : def_RED, 'val' : 2}

globX = 10
globY = 10
is_first_run = True

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = True #False       # flag to check if rect drawn
rect_or_mask = 1 #100     # flag for selecting rect or mask mode
value = def_DRAW_FG         # drawing initialized to FG
thickness = 9           # brush thickness
datasetPtr = None

class Dataset:
    _ix = None
    _iy = None
    #
    mskPrefix = '-msk.png'
    outPrefix = 'processed-images'
    denPrefix = 'denied'
    wdir = None
    pathImgs = None
    cidx = None
    #
    _img = None
    _msk = None
    _img_draw = None
    _rect = None
    _isDrawing = False
    _isRectangle = False
    _is_rect_not_mask = None
    _is_rect_over = False
    _is_drawing = False
    _value = None
    #
    def __init__(self, pdir=None):
        if pdir is not None:
            self.wdir = os.path.abspath(pdir)
            self.updateImages()
    def updateImages(self):
        if (self.wdir is not None):
            tlst = glob.glob('{0}/*.jpg'.format(self.wdir))
            self.pathImgs = np.array(sorted(tlst))
            self.cidx = 0
            self._value = def_DRAW_BG
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
                lstImg = glob.glob('{0}/*.png'.format(self.outputDir()))
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
    # image processing methods
    def _processWithRectangle(self):
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        cv2.grabCut(self._img, self._msk, self._rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
        print(':: grabCut() with rect {0}'.format(self._rect))
        self._is_rect_not_mask = False
    def _processWithMask(self):
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        cv2.grabCut(self._img, self._msk, self._rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_MASK)
    def processImage(self):
        if self._is_rect_not_mask:
            self._processWithRectangle()
            self._is_rect_not_mask = False
        else:
            self._processWithMask()
    #
    def loadCurretImage(self):
        if self.isOk():
            if self.getNumImages()<1:
                print (' !!! WARNING !!! cant find files in directory [{0}], skip...'.format(self.wdir))
                return
            tpathImg = self._getPathImg()
            tpathMsk = self._getPathMsk()
            if os.path.isfile(tpathMsk):
                timg = cv2.imread(tpathMsk, cv2.IMREAD_UNCHANGED)
                self._img = timg[:,:,:3].copy()
                tmsk = (timg[:,:,-1]>250)
                self._is_rect_not_mask = False
                # convert binary mask to Grab-Mask
                self._msk = np.zeros(tmsk.shape, np.uint8)
                self._msk[tmsk > 0] = def_DRAW_PR_FG['val']
                # self._msk[tmsk < 1] = def_DRAW_PR_BG['val']
                self._img_draw = draw_mask_on_image(self._img, self._msk)
            else:
                self._img = cv2.imread(tpathImg)
                # if gray -> convert to RGB
                if self._img.ndim<3:
                    self._img = cv2.cvtColor(self._img, cv2.COLOR_GRAY2BGR)
                self._msk = np.zeros(self._img.shape[:2], np.uint8)
                self._img_draw = self._img.copy()
                self._is_rect_not_mask = True
            #
            tbrd = (def_BBOX_PERCENT * np.array(self._img.shape)).astype(np.int)
            self._rect = (tbrd[1], tbrd[0], self._img.shape[1] - 2 * tbrd[1], self._img.shape[0] - 2 * tbrd[1])
            self.processImage()
            # if self._is_rect_not_mask:
            #     self._processWithRectangle()
    def resetMask(self):
        self._img_draw = self._img.copy()
        self._msk = np.zeros(self._img.shape[:2], np.uint8)
        tpad = 2
        self._rect = (tpad, tpad, self._img.shape[1] - 2*tpad, self._img.shape[0] - 2*tpad)
        self._is_drawing = False
        self._is_rect_not_mask = True
    def saveMasked(self):
        if self.isOk():
            retMsk = np.zeros(self._msk.shape, dtype=np.uint8)
            retMsk[:] = def_MSK_THRESHOLD
            retMsk[np.where((self._msk == 1) + (self._msk == 3))] = 255
            retMasked = np.dstack( (self._img, retMsk) )
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
    def moveDeniedImage(self):
        if self.isOk():
            pathImg = self._getPathImg()
            dirDen = self.deniedDir()
            if os.path.isfile(pathImg):
                shutil.move(pathImg, dirDen)
                self.updateImages()
                self.loadCurretImage()
                print ('DENIED :: MOVE from [{0}] to [{1}]'.format(pathImg, dirDen))
            else:
                print ('\t***Image is not processed!, skip... [{0}]'.format(pathImg))

def draw_mask_on_image(pimg, pmsk):
    pimgR = pimg.copy().reshape(-1,3)
    pmskR = pmsk.reshape(-1)
    # pimgR[pmskR == 0, :] = 0
    pimgR[pmskR == 1, :] = 255
    # probability BG
    pimgR[pmskR == 2, 2] = 255
    # probability FG
    pimgR[pmskR == 3, 1] = 200
    return pimgR.reshape(pimg.shape)

def mark_position(pdataset, px, py, pthickness):
    cv2.circle(pdataset._img_draw, (px, py), pthickness, pdataset._value['color'], -1)
    cv2.circle(pdataset._msk, (px, py), pthickness, pdataset._value['val'], -1)

def draw_all_windows():
    global globX, globY, is_first_run, datasetPtr
    if is_first_run:
        cv2.namedWindow(def_win_input_img, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(def_win_output_msk, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow(def_win_input_img)
        # cv2.namedWindow(def_win_output_msk)
        cv2.setMouseCallback(def_win_input_img, on_mouse)
        cv2.setMouseCallback(def_win_output_msk, on_mouse)
        cv2.moveWindow(def_win_input_img, 600, 50)
        cv2.createTrackbar('Radius', def_win_output_msk, 0, 50, on_track_mouse_size)
        cv2.setTrackbarPos('Radius', def_win_output_msk, thickness)
        is_first_run = False
    if datasetPtr.isOk:
        timg1 = datasetPtr._img_draw.copy()
        # timg2 = datasetPtr._img.copy()
        # (0) prepare masked images
        tmskU8 = np.where((datasetPtr._msk == 1) + (datasetPtr._msk == 3), 255, 0).astype('uint8')
        timg2 = cv2.bitwise_and(datasetPtr._img.copy(), datasetPtr._img.copy(), mask=tmskU8)
        # (1) draw rectangle
        rectP1 = (datasetPtr._rect[0], datasetPtr._rect[1])
        rectP2 = (rectP1[0] + datasetPtr._rect[2], rectP1[1] + datasetPtr._rect[3])
        cv2.rectangle(timg1, rectP1, rectP2, def_BLUE, 2)
        # (2) draw current mouse pointer
        colorCircle = datasetPtr._value['color']
        # cv2.circle(timg1, (globX, globY), thickness, (0, 255, 0))
        # cv2.circle(timg1, (globX, globY), 1, (0, 255, 0))
        # cv2.circle(timg2, (globX, globY), thickness, (0, 255, 0))
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
    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        datasetPtr._isRectangle = True
        datasetPtr._ix = x
        datasetPtr._iy = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if datasetPtr._isRectangle:
            tix = datasetPtr._ix
            tiy = datasetPtr._iy
            trect = (min(tix,x),min(tiy,y),abs(tix-x),abs(tiy-y))
            datasetPtr._rect = trect
            datasetPtr._is_rect_not_mask = True
    elif event == cv2.EVENT_RBUTTONUP:
        datasetPtr._isRectangle = False
        datasetPtr._is_rect_over = True
        tix = datasetPtr._ix
        tiy = datasetPtr._iy
        trect = (min(tix, x), min(tiy, y), abs(tix - x), abs(tiy - y))
        datasetPtr._rect = trect
        datasetPtr._is_rect_not_mask = True
        print(" Now press the key 'n' a few times until no further change \n")
    # draw touchup curves
    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
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

def on_track_mouse_size(x):
    global thickness
    thickness = x +1

if __name__ == '__main__':
    if os.path.isdir(def_WORK_DIR):
        print ('**WARNING** found default directory.\n\tImages will be loaded from this default directory [{0}]'.format(def_WORK_DIR))
        datasetPtr = Dataset(pdir=def_WORK_DIR)
    else:
        if len(sys.argv) < 2:
            print ('Usage: {0} {{/path/to/dir/with/images}}'.format(sys.argv[0]))
            sys.exit(0)
        else:
            datasetPtr = Dataset(pdir=sys.argv[1])

    print (datasetPtr)
    datasetPtr.loadCurretImage()
    draw_all_windows()

    while True:
        k = cv2.waitKey()
        # key bindings
        if k == 27:  # esc to exit
            break
        elif k == ord('0'):  # BG drawing
            print(" mark background regions with left mouse button \n")
            datasetPtr._value = def_DRAW_BG
        elif k == ord('1'):  # FG drawing
            print(" mark foreground regions with left mouse button \n")
            datasetPtr._value = def_DRAW_FG
        elif k == ord('2'):  # PR_BG drawing
            datasetPtr._value = def_DRAW_PR_BG
        elif k == ord('3'):  # PR_FG drawing
            datasetPtr._value = def_DRAW_PR_FG
        elif k == ord('r'):
            datasetPtr.resetMask()
        elif k == ord('n'):
            datasetPtr.processImage()
        elif k == ord('k'):
            if datasetPtr.prevImageIdx():
                datasetPtr.loadCurretImage()
            else:
                print ('!!! Cannt load previous image: {0}'.format(datasetPtr._toString()))
        elif k == ord('l'):
            if datasetPtr.nextImageIdx():
                datasetPtr.loadCurretImage()
            else:
                print ('!!! Cannt load next image: {0}'.format(datasetPtr._toString()))
        elif k == ord('s'):
            datasetPtr.saveMasked()
        elif k == ord('m'):
            datasetPtr.moveProcessedImage()
        elif k == ord('d'):
            datasetPtr.moveDeniedImage()
        elif k == ord('h'):
            print (__doc__)

        draw_all_windows()

