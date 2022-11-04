#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 24 14:44:40 2020

@author: Williams modified by Lionel 2/7/2020

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
op = os.path
opd, opb, opj = op.dirname, op.basename, op.join
opa = op.abspath
from time import time
import yaml
from pathlib import Path
from colorama import Fore, Back, Style      # Color in the Terminal
from matplotlib import pyplot as plt
import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle
from tensorflow.keras import models
from modules_unet.util_predict import UTIL
from modules_unet.handle_images import HANDLE


class MAKE_PREDICTION():
    '''

    '''
    def __init__(self):
        '''
        '''

    def pred(self, test, model_name, model_loaded, nb_denoising=0, debug=[0,1,2]):
        '''
        test: folder where are the pictures
        model_name: model used name
        model_loaded : model loaded
        file:
        meth: 'unet' or 'stardist'
        '''
        if 0 in debug:
            print(f'model name is {model_name}')
            try:
                print(f'model alias is {model_loaded.alias}')
            except:
                print('No associated alias ')
        self.ut = UTIL(test, model_name)
        dir_test_images = Path('test') / self.ut.test
        if model_loaded.mod_type == 'unet':
            gray=True if model_loaded.input_shape[3] == 1  else False                         # image with 3 or 1 level
        else:
            gray=False
        ha = HANDLE(dir_test_images, kind='test', dim=512, gray=gray)
        self.ut.make_predict_subdir(ha)                                                            # subdir for the predictions
        meth = self.seg_meth(model_loaded.alias)                                                 # find the kind of method from the 2 first letters

        for i,test_im in enumerate(ha.tab_test_images):
            t0 = time()
            if nb_denoising: test_im = self.denoising(test_im, nb_denoising=nb_denoising)          # denoising
            if meth == 'unet':
                prediction = model_loaded.predict(np.array([test_im]))                   # make prediction with Unet
            elif meth == 'stardist':
                img_modif = self.rescale(np.array([test_im])[0,:,:,0], fact=1)
                pred_sd, _ = model_loaded.predict_instances(img_modif)                      # make prediction with Stardist
                prediction = [self.rescale(pred_sd, fact=1)]
                # if 1 in debug:
                #     plt.imshow(prediction[0]*255)
                #     plt.savefig(opj('temp', f'pred_{i}.png'))
            t1 = time()
            if 2 in debug:
                telapsed = round(t1-t0, 3)
                print(f'time for prediction is {telapsed} sec')
            self.ut.save_prediction(i, prediction, meth)

        print(Fore.YELLOW + '########  Predictions done')
        print(Style.RESET_ALL)

    def seg_meth(self, model_name, debug=[0]):
        '''
        Kind of methohd from the first two letters
        '''
        if model_name[:2] == 'Sd':
            meth = 'stardist'
        else:
            meth = 'unet'
        if 0 in debug:
            print(f'The segmentation method used is {meth} ')

        return meth

    def rescale(self, img, fact=255):
        '''
        '''
        img = (img-img.min())/(img.max()-img.min())*fact

        return img

    def denoising(self, img, kind='fastNlMeans', nb_denoising=0):
        '''
        Denoising
        '''
        #for i in range(int(self.args.nb_denoising)):
        for i in range(nb_denoising):
            if kind == 'fastNlMeans':
                img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            else:
                img = denoise_tv_chambolle(img, weight=0.03)

        return img
