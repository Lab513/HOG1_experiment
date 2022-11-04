import os
import json
import shutil as sh
from datetime import datetime
from pathlib import Path
import pickle as pkl
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
from scipy.linalg import norm

class FLUO():
    '''
    '''
    def __init__(self):
        pass

    def func_fluo(self, c, i, func, col, debug=[]):
        '''
        Generic function for fluo applied in the segmented contour..
        c : contour
        i : image index
        func : function called (sum, std etc..)
        col : fluo color asked
        '''
        mask = self.mask_from_cntr(c)
        list_fluo = getattr(self, f'list_imgs_fluo{col}')
        if 1 in debug:
            print(f'## len(list_fluo) is {len(list_fluo)}')
            print(f'i is {i}')
            print(f'type(list_fluo[0]) is {type(list_fluo[0])}')
        if func in ['sum', 'std']:
            val = getattr(list_fluo[i][mask == 255], func)()                  # operation on fluo image in the contour mask..
        else :
            fluo_px = list_fluo[i][mask == 255]
            if func == 'contrast':
                fmin,fmax = int(fluo_px.min()), int(fluo_px.max())
                #if 3 in debug:
                print(f' fmin {fmin}')
                print(f' fmax {fmax}')
                # print(f' type(fmax) {type(fmax)}')
                # print(f' type(fmin) {type(fmin)}')
                val = (fmax - fmin)/(fmax + fmin)
                print(f'(fmax - fmin)/(fmax + fmin), val is {(fmax - fmin)/(fmax + fmin)}')
                print(f'In func_fluo, val is {val}')
        if 2 in debug:
            print(f'func is {func}')
            print(f'In func_fluo, val is {val}')

        return val

    def std_fluo(self,c,i,col,norm=False):
        '''
        Standard deviation of the fluorescence using the segmentation mask in one cell
        c : contour
        i : image index
        col : fluo color asked
        norm = normalization factor, using contour area
        '''
        norm = cv2.contourArea(c) if norm else 1
        return self.func_fluo(c,i,'std')/norm

    def sum_fluo(self, c, i, col, norm=False, debug=[]):
        '''
        Integration of the fluorescence using the segmentation mask  in one cell
        c : contour
        i : image index
        col : fluo color asked
        norm = normalization factor, using contour area
        '''
        if type(c) == np.ndarray :                                                # case c = None
            area = cv2.contourArea(c) if norm else 1
        else:
            return 0
        area = 1 if area == 0  else area                                        # protection against null areas..
        if 1 in debug: print(f"area is {area}")
        try:
            return self.func_fluo(c, i, 'sum', col)/area
        except:
            return 0            # non existing value, cell not yet created

    def contrast_fluo(self, c, i, col, norm=False, debug=[0]):
        '''
        Contrast of the fluorescence using the segmentation mask  in one cell
        c : contour
        i : image index
        col : fluo color asked
        norm = normalization factor, using contour area
        '''
        if 0 in debug: print('Calling contrast_fluo !!!')
        area = cv2.contourArea(c) if norm else 1
        area = 1 if area == 0  else area                                        # protection against null areas..
        if 1 in debug: print(f"area is {area}")
        return self.func_fluo(c,i,'contrast',col)
