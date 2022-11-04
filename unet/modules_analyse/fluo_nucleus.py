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

class FLUO_NUCLEUS():
    '''
    '''
    def __init__(self):
        pass

    def find_contours_fluo(self, img, imnum, debug=[]):
        '''
        Contours obtain with RFP images..
        '''
        img = self.cf.bytescaling(img)
        if 2 in debug:
            cv2.imwrite(f'fluo_bscale_{imnum}.png', img)
        img = cv2.GaussianBlur(img,(5,5),0)
        if 3 in debug:
            cv2.imwrite(f'fluo_gaubl_{imnum}.png', img)
        #img = self.cf.filter2D(img)                                                          # filtering with kernel..
        # if 3 in debug:
        #     cv2.imwrite(f'fluo_kern_{imnum}.png', img)
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.cntrs_fluo, _ = cv2.findContours(th2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if 0 in debug:
            print(f' len(self.cntrs_fluo) {len(self.cntrs_fluo)}')
        if 1 in debug:
            cv2.imwrite(f'fluo_{imnum}.png', img)
            cv2.imwrite('fluo_thresh.png', th2)

    def fluo_positions(self):
        '''
        in one fluo image make list of the positions..
        '''
        self.positions_fluo = []
        for c in self.cntrs_fluo:
            self.positions_fluo.append(self.pos(c))

    def find_nearest_fluo_contour(self, im_num, debug=[]):
        '''
        Find nearest contour for tracking in fluo
        '''
        if 0 in debug: print(f'self.id, im_num are {self.id, im_num}')
        pos_fluo = np.array(self.positions_fluo)                              # all fluo positions in image im_num
        pos_segm = self.positions[self.id][im_num]                              # pos of cell self.id in image im_num
        if 1 in debug: print(f'pos_fluo {pos_fluo}')
        if 2 in debug: print(f'pos_segm {pos_segm}')
        list_distances = list(map(norm, (pos_fluo - np.array(pos_segm))))
        self.ind_fluo_nearest = np.argmin(list_distances)
        if 3 in debug:
            print(f"index of fluo contour nearest from cell {self.id} is {self.ind_fluo_nearest}")

    def nearest_fluo_index(self, im_num, debug=[]):
        '''
        Find index of the nearest contour
        '''
        if 0 in debug:
            print(f'***** im_num is {im_num} !!!!')
        img = self.list_imgs_fluo1[im_num]
        if 1 in debug:
            cv2.imwrite(f'fluo_orig_{im_num}.png',img)
        self.find_contours_fluo(img, im_num)
        self.fluo_positions()
        self.find_nearest_fluo_contour(im_num)

    def nearest_fluo_contour(self, im_num, debug=[]):
        '''
        Find the nearest fluo contour
        '''
        self.nearest_fluo_index(im_num)
        if 0 in debug: print(f'self.cntrs_fluo is {self.cntrs_fluo} ')
        nearest_fluo_cntr = self.cntrs_fluo[self.ind_fluo_nearest]
        #mask = self.self.mask_from_cntr(nearest_fluo_cntr)

        return nearest_fluo_cntr

    def func_fluo_in_nucleus(self,i,func,debug=[]):
        '''
        Generic function for fluo applied in the segmented contour..
        i : image index
        using self.list_imgs_fluo2
        '''
        c = self.nearest_fluo_contour(i)
        if 0 in debug: print(f'############## contour c is {c} ')
        mask = self.mask_from_cntr(c)
        if 1 in debug: print(f'mask {mask} ')
        val = getattr(self.list_imgs_fluo2[i][mask > 200], func)()           # operation on fluo image in the contour mask..
        return val

    def fluo_nucleus_sequence(self, func, debug=[0]):
        '''
        '''
        self.fluo_nucl = []                                                       # current observation
        if 0 in debug:
            print(f'len(self.list_cntrs_ci) {len(self.list_cntrs_ci)}')
            print(f'len(self.list_imgs_BF) {len(self.list_imgs_BF)}')
        for i in range(len(self.cntrs)):
            val_nucl_fluo = self.func_fluo_in_nucleus(i,func)                     # return the value of the fluo in the nucleus.
            self.fluo_nucl += [val_nucl_fluo]

    def fluo_nucl_mean(self, debug=[]):
        '''
        Mean fluo in the nucleus..
        '''
        list_mean = []
        for i in range(len(self.cntrs)):
            try:
                ## nucleus
                img_nucl = self.list_imgs_fluo1[i]
                c = self.nearest_fluo_contour(i)                                              # RFP contour
                mask0 = self.mask_from_cntr(c)                                                # RFP mask
                nucl_mean = self.list_imgs_fluo2[i][mask0 > 200].mean()                       # mean signal in the nucleus
                list_mean += [nucl_mean]
            except:
                print('Cannot calculate the mean for the nucleus..')
            if 1 in debug:
                print(f'list_mean = {list_mean}')

        return list_mean

    def fluo_ratio_with_nucl(self, kind='whole_cell'):
        '''
        Coefficient for colocalization of GFP in the nucleus for a given cell
        kind: 'whole_cell' or 'cyto'
        keywords: Colocalization calculus, colocalization coeff, colocalisation
        '''
        list_ratios = []
        for i in range(len(self.cntrs)):
            try:
                #self.list_imgs_fluo2[i]
                ## nucleus
                img_nucl = self.list_imgs_fluo1[i]
                c = self.nearest_fluo_contour(i)                                             # RFP contour
                mask0 = self.mask_from_cntr(c)                                               # RFP mask
                nucl = self.list_imgs_fluo2[i][mask0 > 200].mean()                           # mean signal in the nucleus
                c = self.list_cntrs_ci[i]                                                    # BF contour
                mask1 = self.mask_from_cntr(c)                                               # cell contour mask from BF segm
                if kind == 'cyto':
                    # cytoplasm = whole cell - nucleus
                    cyto = self.list_imgs_fluo2[i][(mask1 > 200) & (mask0 < 10)].mean()      # mean signal in the cytoplasm
                    list_ratios += [nucl/cyto]                                               # ratio nucl/cytoplasm
                elif kind == 'whole_cell':
                    whole = self.list_imgs_fluo2[i][(mask1 > 200)].mean()                    # mean signal in the whole cell
                    list_ratios += [nucl/whole]                                              # ratio nucl/whole cell
            except:
                # The cell does not exist yet !!!
                list_ratios += [None]

        return list_ratios

    def fluo_in_nucleus(self, kind='ratio', norm=True, raw=True, debug=[]):
        '''
        Plot the ratio fluo in nucleus / fluo in the cell
        kind: ratio, mean_nucl, contrast
        '''
        if 0 in debug: print(f'self.id is {self.id}')

        if kind == 'ratio':
            self.curr_obs = self.fluo_ratio_with_nucl()
            if 1 in debug: plt.show()
        if kind == 'mean_nucl':
            self.curr_obs = self.fluo_nucl_mean()
            if 1 in debug: plt.show()
        if kind == 'contrast':
            plt.title('contrast')
            self.fluos('contrast', 2, norm=True).plot()
        ###
        self.obs = 'nucl_fluo'
        # Title
        if kind == 'mean_nucl':
            self.title = 'mean nucleus fluo'
        else:
            self.title = 'colocalization'

        return self

    def sum_yellow_fluo(self, c,i):
        '''
        Integration of the fluorescence using the segmentation mask
        c : contour
        i : image index
        '''
        return self.func_fluo_in_nucleus(i,'sum')
