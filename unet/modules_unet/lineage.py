import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
op = os.path
opb, opd, opj = op.basename, op.dirname, op.join
from matplotlib import pyplot as plt
import numpy as np
import cv2
from skimage import io
from tensorflow.keras import models
from colorama import Fore, Back, Style
##
from color_and_id import COLOR_AND_ID as CI



class LINEAGE(CI):
    '''
    '''
    def __init__(self, root_addr=None):
        '''
        '''
        CI.__init__(self)
        self.nb_assoc_tot = 0
        self.delta_cells = 0
        self.all_assoc = []
        self.dic_modo = {}
        self.l_assoc_tot = []
        self.l_delta_cells = []

    def find_contours_anaph(self, pred, thr=127, size_min=None, debug=[0,1]):
        '''
        Find the contours on RFP prediction
        size_min : minimal size along the max ellipse axis which fits the shape
        '''
        if 0 in debug:
            print(f'pred.shape is {pred.shape} ')
        ret, thresh = cv2.threshold(pred[:,:,0].astype('uint8'), thr, 255, 0)
        cntrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if 1 in debug:
            print(f'len(cntrs) is {len(cntrs)} ')
        if size_min:
            lcntrs = []
            for c in cntrs:
                try:
                    ellipse = cv2.fitEllipse(c)          # fit an ellipse on the anaphase shape..
                    (xc,yc),(d1,d2),angle = ellipse
                    d2 = round(d2,1)
                    print(f'd2 is {d2} !!!')
                    if d2 > size_min:
                        lcntrs += [c]
                except:
                    pass
            cntrs = lcntrs

        return cntrs

    def make_mask_from_contour(self, cnt):
        '''
        Mask from contour for finding intersections
        '''
        h, w, nchan = (512, 512, 1)
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1) # fill contour

        return mask

    def image_contours(self, list_cntrs):
        '''
        Mask from contour for finding intersections
        list_cntrs : list of the contours
        '''
        h, w, nchan = (512,512,1)
        img = np.zeros((h, w), np.uint8)
        cv2.drawContours(img, list_cntrs, -1, (255, 255, 255), -1) # fill contour

        return img

    def find_mother_daughter(self, l_intsc, list_cntrs, debug=[]):
        '''
        Between two cells find which is the mother and which is the daughter..
        l_intsc : list of two cells or less
        list_cntrs : list of all the cells contours
        '''
        max_area = 0
        for i,ind in enumerate(l_intsc):
            cnt = list_cntrs[ind]
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                mother =  l_intsc[i]
                daughter = l_intsc[abs(i-1)]
            if 1 in debug:
                img_mo_do = self.image_contours([cnt])
                plt.imshow(img_mo_do)
                plt.show()

        print(Fore.YELLOW + f'**** mother is {mother} and daughter {daughter} ****')
        print(Style.RESET_ALL)
        modo = str(mother) + '_' + str(daughter)

        if mother in self.dic_modo.keys():
            self.dic_modo[mother] += [daughter]  # add
        else:
            self.dic_modo[mother] = [daughter]   # init
        if modo not in self.all_assoc:
            self.all_assoc += [modo]
            self.nb_assoc += 1
        self.rand_col_lineage[daughter] = self.rand_col_lineage[mother]                    # daughter inherit color from mother

    def find_intersect_cells(self, c, list_cntrs, debug=[]):
        '''
        Compare Intersection over Union..
        return cell num
        c : contours for anaphase
        list_cntrs : contours for cells
        '''
        if 0 in debug:
            print(f'len(list_cntrs) is {len(list_cntrs)}')
        mask0 = self.make_mask_from_contour(c)                    # anaphase
        l_intsc = []                                                # init list intersections
        for i,c1 in enumerate(list_cntrs):
            mask1 = self.make_mask_from_contour(c1)               # cells contours
            inter = np.logical_and(mask0, mask1)
            if inter.any():
                l_intsc += [i]
        if l_intsc:
            if len(l_intsc) == 2:
                if 2 in debug: print(f'Intersection with cells {l_intsc} ')
                self.find_mother_daughter(l_intsc, list_cntrs)
            else:
                l_intsc = []
                if 3 in debug: print(f'No intersection found ')
            return l_intsc
        else:
            return None

    def prepare_for_association(self, num_pic=None,
                                      thr = 120):
        '''
        Load the pictures, make the predictions, extract the contours
        before associating mother and daughter cells..
        '''
        self.nb_assoc = 0
        self.thr = thr
        self.load_fluo(num_pic)
        self.extract_cntr_anaph()

    def load_fluo(self, num, debug=[0]):
        '''
        Load the fluo prediction for image num
        '''
        if 0 in debug: print(f'self.ut.path_pred_anaph is {self.ut.path_pred_anaph}')
        self.img_pred_fluo = cv2.imread(opj(self.ut.path_pred_anaph, f'frame{num}.png'))

    def make_pic_fluo_cntrs(self):
        '''
        '''
        h, w, nchan = (512,512,1)
        self.img_anaph = np.zeros((h, w), np.uint8)
        cv2.drawContours(self.img_anaph , self.cntrs_anaph, -1, (255, 255, 255), -1) # fill contour
        if self.show_fluo_cntrs:
            plt.imshow(self.img_anaph)
            plt.figure()

    def extract_cntr_anaph(self, debug=[1]):
        '''
        Extract the contours from the predictions on RFP images
        Create a list of the contours for the anaphase : self.cntrs_anaph
        '''
        self.cntrs_anaph = self.find_contours_anaph(self.img_pred_fluo, thr=self.thr, size_min=5)
        if 1 in debug: print(f'len(self.cntrs_anaph) is {len(self.cntrs_anaph)}')
        self.make_pic_fluo_cntrs()

    def find_associations(self, num_anaph=None, debug=[1]):
        '''
        For a given anaphase detection, find the two corresponding mother and daughter cells
        num_anaph : index of the anaphase contour
        '''
        # try:
        print(f'num_anaph is {num_anaph}')
        self.c_anaph = self.cntrs_anaph[num_anaph]                              # contour anaphase selected
        self.l_intersc = self.find_intersect_cells(self.c_anaph, self.curr_contours)       # list of associated cells
        if self.l_intersc:
            self.list_anaph_ret += [self.c_anaph]                             # retain the contour
        if 1 in debug:
            print(f'self.l_intersc is {self.l_intersc}')                    # cells with filiation
        print('find intersect finished..')

        # except:
        #     if 2 in debug:
        #         print('num_anaph probably does not exist.. ')

    def make_lineage(self, num_pic, thr=120,
                                          show_fluo=False,
                                          show_pred_fluo=False,
                                          show_fluo_cntrs=True,
                                          show_cntrs_ret=False,
                                          show_cntrs_and_BF=False,
                                          show_cntrs_ret_and_BF=False,
                                          debug=[0]):
        '''
        Find all the associatons in one picture
        num_pic : picture index
        thr : threshold
        '''
        self.list_anaph_ret = []
        self.show_fluo = show_fluo
        self.show_pred_fluo = show_pred_fluo
        self.show_fluo_cntrs = show_fluo_cntrs
        self.show_cntrs_ret = show_cntrs_ret
        self.show_cntrs_and_BF = show_cntrs_and_BF
        self.show_cntrs_ret_and_BF = show_cntrs_ret_and_BF
        self.prepare_for_association(num_pic=num_pic, thr=thr)
        if num_pic == 0:
            self.nb_cells0 = len(self.curr_contours)
        self.delta_cells = len(self.curr_contours) -  self.nb_cells0
        for i,num_anaph in enumerate(range(len(self.cntrs_anaph))):
            if 0 in debug:
                print(f'**** Searching intersection with anaphase contour {i}')
            self.find_associations(num_anaph=num_anaph)
        self.color_and_id_for_cells(self.img_lineage, self.rand_col_lineage)                    # color and Id tracking number on self.img
        self.nb_assoc_tot += self.nb_assoc
        print(Fore.GREEN + f' In current picture, found {self.nb_assoc} associated cells')
        print(f' Until pic num {num_pic}, found {self.nb_assoc_tot} associations !!!')
        print(f'delta cells is {self.delta_cells}')
        print(Style.RESET_ALL)
        #self.show_cntrs_in_diff_situations()

    def show_cntrs_in_diff_situations(self):
        '''
        '''
        self.img_anaph_ret = self.image_contours(self.list_anaph_ret)                     # image of retained cntrs
        if self.show_cntrs_ret:
            plt.imshow(self.img_anaph_ret)
            plt.figure()
        self.img_bf = self.img_bf[0,:,:,0]
        self.img_bf = (self.img_bf/self.img_bf.max()*255).astype('uint8')
        if self.show_cntrs_and_BF:
            self.superp_cntrs_BF()
        if self.show_cntrs_ret_and_BF:
            self.superp_cntrs_ret_BF()

    def superp_cntrs_BF(self):
        '''
        Show cntrs with BF
        '''
        self.superp = cv2.addWeighted(self.img_anaph, 0.2, self.img_bf, 0.7, 0)
        plt.imshow(self.superp)
        plt.figure()

    def superp_cntrs_ret_BF(self):
        '''
        Show retained cntrs with BF
        '''
        self.superp_ret = cv2.addWeighted(self.img_anaph_ret, 0.2, self.img_bf, 0.7, 0)
        plt.imshow(self.superp_ret)
        plt.figure()
