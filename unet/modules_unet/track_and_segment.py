'''
py detect_cells.py -f '/home/meglio/Bureau/boulot/cells_videos/BF_f0000.tif'  -m  ep5_dil3_fl  --track all
'''
import os, sys
op = os.path
opb, opd, opj = op.basename, op.dirname, op.join
from pathlib import Path
import glob
import copy
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import norm
from colorama import Fore, Back, Style
from events import EVENTS as EV
from track_methods import TRACK_METH as TM
from track_misc import TRACK_MISC as TMI
from track_with_btrack import APPLY_BTRACK as AB
from color_and_id import COLOR_AND_ID as CI

class TRACK_SEGM(EV,TM,AB,CI,TMI):
    '''
    Tracking and Segmentation
    '''

    def __init__(self):
        '''
        '''
        ldep = [EV,TM,AB,CI,TMI]
        [dep.__init__(self) for dep in ldep]    # init for all the inheritances
        # Tracking
        self.iou_score_low_lim = 0.3           # inter over union score low limit
        self.list_prev_pos = []                # empty at the beginning
        self.pos_at_frame = []

    def random_colors(self):
        '''
        create random colors
        '''
        min_col = 254 if self.args.one_color else 0
        self.rand_col = [tuple(map(int,np.random.randint(min_col,255,3))) for i in range(20000)]  # random color for tracking
        self.rand_col_lineage = self.rand_col.copy()

    def fit_ellipse(self,c):
        '''
        Make the list of ellipses
        '''
        try:
            ell = cv2.fitEllipse(c)
            self.list_ellipses.append(ell)
        except:
            self.list_ellipses.append('a')

    def fill_list_pos(self, c, show_pos=False):
        '''
        Make the list of positions
        '''
        x, y, w, h = cv2.boundingRect(c)
        pos = (int(x + w/2), int(y + h/2)) # find position
        self.list_pos.append(pos)
        if show_pos:
            self.circle_at_pos(pos)

    def fill_list_pos_events(self, c, show_pos=False):
        '''
        Make the list of the events positions
        '''
        x, y, w, h = cv2.boundingRect(c)
        pos = (int(x + w/2), int(y + h/2))
        self.list_pos_events.append(pos)
        if show_pos:
            self.circle_at_pos(pos)

    def tracking_with_btrack(self, contours):
        '''
        '''
        for i,pos in enumerate(self.list_pos):                                  # list for btrack
            self.pos_at_frame.append({"t": self.num,
                                        "x": pos[0] ,
                                        "y": pos[1] ,
                                        "Contours": contours[i]})
        self.pos_per_frm = pd.DataFrame(self.pos_at_frame)
        self.find_tracks_btrack()                                               # make the tracks with Btrack

    def find_position_from_contour(self, all_cntrs, debug=[]):
        '''
        For each cell prediction, find the position from the bounding rectangle
        '''
        contours, contours_events = all_cntrs
        self.list_pos = []
        self.list_ellipses = []
        self.curr_contours = self.contours[self.num] = contours
        for c in contours:
            if 0 in debug:
                print(f'len(c) is {len(c)} ')
                print(f'type(c) is {type(c)} ')
                print(f'c[:5] is {c[:5]}')
            self.fill_list_pos(c)                                               # create the list of positions
            self.fit_ellipse(c)
        if contours_events:
            self.list_pos_events = []
            self.curr_contours_events = self.contours_events[self.num] = contours_events
            for c in contours_events:
                self.fill_list_pos_events(c)                                    # create the list of events positions
        ## Find tracks using btrack
        if self.args.kind_track == 'btrack':
            self.tracking_with_btrack(contours)

    def swap_contours(self,i,ind):
        '''
        Change the index of the contour
        '''
        buff = self.curr_contours[i]
        self.curr_contours[i] = self.curr_contours[ind]
        self.curr_contours[ind] = buff                       # swapped contours

    def swap_positions(self,i,ind):
        '''
        Change the index of the pos
        '''
        buff = self.list_pos[i]
        self.list_pos[i] = self.list_pos[ind]
        self.list_pos[ind] = buff                            # swapped positions

    def make_swaps(self,i,ind):
        '''
        Swap the positions and the mask
        '''
        #print('list_distances[ind] is ', list_distances[ind])
        if i not in self.list_indices_pos:
            print(Fore.RED + Style.NORMAL + f'swapping ### i: {i} and ind: {ind} ')
            print(Style.RESET_ALL)
            self.swap_positions(i,ind)
            self.swap_contours(i,ind)                                           # refresh the index of the contour previously indexed i
            self.list_indices_pos.append(i)                                     # block position i
        print(Fore.BLUE + str(self.list_indices_pos))

    def dilate_mask(self, mask, dil=1, iter_dil=1):
        '''
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil,dil))
        mask = cv2.dilate(mask, kernel, iterations = iter_dil)    # dilate
        return mask

    def make_mask_from_contour(self,cnt, dilate=False, iter_dilate=1):
        '''
        '''

        h, w, nchan = self.img.shape
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1) # fill contour
        if dilate:
            mask = self.dilate_mask(mask, dil=dilate, iter_dil=iter_dilate)
        return mask

    def compare_areas(self,i,ind):
        '''
        Compare areas of the contours i and ind
        '''
        area0 = cv2.contourArea(self.contours[self.num-1][i])  # previous contour i
        area1 = cv2.contourArea(self.curr_contours[ind])       # contour ind in new list
        ratio_areas = area0/area1

        return ratio_areas

    def IoU_filter(self,i,ind):
        '''
        Compare Intersection over Union..
        '''

        mask0 = self.make_mask_from_contour(self.contours[self.num-1][i])                      # previous contour
        mask1 = self.make_mask_from_contour(self.curr_contours[ind])                           # new contour..
        inter = np.logical_and(mask0, mask1)
        union = np.logical_or(mask0, mask1)
        iou_score = np.sum(inter) / np.sum(union)
        print("### IoU score for {0} with {1} is {2}  ".format(i, ind, iou_score))
        return iou_score

    def changing_tests(self,i,ind):
        '''
        '''

        iou_score = self.IoU_filter(i,ind)                                         # comparing intersection over union
        if iou_score < self.iou_score_low_lim:
            print(Fore.YELLOW + Style.NORMAL +\
                  f'#### Huge difference between {i} and {ind} !!!! ')
            print(Style.RESET_ALL)
        ratio_areas = self.compare_areas(i,ind)                                    # compare the areas of the contours
        if 0.7 < ratio_areas < 1.5:
            print(Fore.YELLOW + Style.NORMAL +\
                   f'#### size ratio between {i} and {ind} is normal')
            print(Style.RESET_ALL)
        else :
            diff = 'from {0} to {1} by factor {2} '.format(i, ind, ratio_areas)
            if ratio_areas < 1:
                print(Fore.RED + Style.NORMAL + '#### size increased ' + diff)
            else:
                print(Fore.RED + Style.NORMAL + '#### size decreased ' + diff)
            print(Style.RESET_ALL)

    def find_nearest_index_track(self,i):
        '''
        Index in the old position list of the nearest position of index i in the new position list
        '''

        return self.find_nearest_index(i, self.list_prev_pos)

    def radius_times_crit(self, i, ind, nbradius=2):
        '''
        Radius times criterion
        '''
        cnt0 = self.contours[self.num-1][i]                                      # contour of previous picture
        r0 = np.sqrt(cv2.contourArea(cnt0)/np.pi)                            # radius of the original contour
        return self.list_distances[ind] < nbradius*r0                            # new distance must be less than 2 times the radius..

    def change_with_nearest(self, i, dist_max=False):
        '''
        Change the position and the contours with the nearest elemt from i ..
        '''
        print('dealing with ', i)
        ind = self.find_nearest_index_track(i)                                   # index of the nearest contour from i
        self.changing_tests(i,ind)                                             # detect if change is normal or not
        if dist_max:
            if self.radius_times_crit(i, ind):                                   # new distance must be less than 2 times the radius..
                self.make_swaps(i,ind)                                         # swap both masks and positions
        else:
            self.make_swaps(i,ind)

    def redefine_Id(self, meth='min', debug=[0]):
        '''
        Refresh each position with the nearest one..
        '''
        if 0 in debug:
            print(f'In redefine_Id !!!')
        if meth == 'min':
            self.meth_min(self.list_prev_pos, self.list_pos, corr=True)
        elif meth == 'hung':
            self.meth_hung(self.list_prev_pos, self.list_pos, corr=True)
        elif meth == 'no_track':
            pass

    def cell_tracking_and_segmentation(self, num, all_cntrs,
                                        show_pos=False, debug=[]):
        '''
        Track the cells
        num : index of the image
        '''
        self.img_lineage = self.img.copy()                                 # copy the BF for lineage
        self.num = num
        self.find_position_from_contour(all_cntrs)                       # listpos
        if 0 in debug: self.track_mess0()                                  # message at beginning
        if self.list_prev_pos:
            old_pos = copy.deepcopy(self.list_pos)
            self.redefine_Id(meth=self.args.kind_track)                  # tracking
            new_pos = copy.deepcopy(self.list_pos)
            self.test_out_track_algo(old_pos, new_pos)
            if show_pos:
                self.show_new_positions()
        self.find_indices_events()                                              # events like buds
        self.color_and_id_for_cells(self.img, self.rand_col)                    # color and Id tracking number on self.img
        self.list_prev_pos = self.list_pos                                      # save positions for tracking
        if 1 in debug: self.track_mess1()                                       # list_prev_pos
