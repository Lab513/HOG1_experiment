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

class SINGLE_CELL():
    '''
    Analysis for single cell
    '''
    def __init__(self):
        pass

    def cell(self,i):
        '''
        Select a cell
        i : cell id
        '''
        self.id = i                                                             # identity of the cell
        self.list_cntrs_ci = []                                                 # list of contours for cell i
        for j,lc in self.cntrs.items():                                         # go through the images, lc : list_contours for current image
            try:
                if type(lc[i]) == np.ndarray:                                   # bt tracks
                    self.list_cntrs_ci.append(lc[i])                          # add contour cell i
                else:
                    self.list_cntrs_ci.append(0)                                # no contour at this time
            except:
                self.list_cntrs_ci.append(0)                                    # no contour at this time
        self.BF_positions()

        return self

    def make_list_areas(self, l):
        '''
        l : list of the contours for a specific cell during time
        '''
        list_areas = []
        for c in l:                                                             # contour for unique cell during time
            try:
                list_areas.append(cv2.contourArea(c))                         # append the surface in function of the time
            except:
                list_areas.append(0)                                            # no contour at this time

        return list_areas                                                       # list of areas during time for unique cell

    def areas(self, title=None):
        '''
        list of the areas of the cell during time
        '''
        self.obs = 'area'
        list_areas = self.make_list_areas(self.list_cntrs_ci)
        self.curr_obs = list_areas
        if not title:
            self.title = f'area evolution for cell {self.id}'
        else:
            self.title = title
        self.xlabel = 'frames'
        self.ylabel = 'cell area'

        return self

    def BF_positions(self, debug=[]):
        '''
        '''
        list_pos_ci = []                                                        # list of the position of the cell i
        self.positions = {}
        for c in self.list_cntrs_ci:
            try:
                list_pos_ci.append(self.pos(c))                               # append the position in function of the time
            except:
                if 1 in debug: print('position not existing')
        x, y = [p[0] for p in list_pos_ci], [p[1] for p in list_pos_ci]
        self.positions[self.id] = list_pos_ci

    def track(self, *args, debug=[]):
        '''
        make the list of the positions of the cell during the time
        '''
        self.obs = 'track'
        self.BF_positions()
        for arg in args:
            if arg == 'corr': x,y = self.cut_at_big_jump(x,y)                   #  cut the tracking when there is a "big jump"
        self.title = 'cell tracking'
        self.curr_obs = [x,y]

        return self

    def find_steps(self):
        '''
        '''
        self.steps = list(map(norm, np.diff(self.positions[self.id], axis=0)))
        return self

    def find_if_big_jumps(self,big_jump=50):
        '''
        Detect a big jump in the track
        '''
        self.find_steps()

        return True if max(self.steps) > big_jump else False

    def step_track(self):
        '''
        list of the distances between the tracking steps..
        '''
        self.find_steps()
        self.steps_filtered = []
        for s in self.steps:
            self.steps_filtered += [s]
            if s > self.lim_step_max:
                break
        print("len(self.steps) ", len(self.steps))
        print("len(self.steps_filtered) ", len(self.steps_filtered))
        self.len_no_jump = len(self.steps_filtered)

    def cut_at_big_jump(self,x,y):
        '''
        Cut x and y at the first big jump
        A posteriori correction
        '''
        self.step_track()
        s = slice(self.len_no_jump)

        return x[s],y[s]

    def list_correct_tracks(self):
        '''
        Save the track while no big jump..
        '''
        self.count()
        max_nb_cells = self.nb_cells[len(self.cntrs)-1]
        self.correct_tracks = []
        for i in range(216):
            try:
                if not self.cell(i).track().find_if_big_jumps():
                    self.correct_tracks += [i]
            except:
                pass

    def fluo_val(self,c,i,col,norm, debug=[]):
        '''
        Fluo value
        c : current cell
        i : frame index
        col : 1 for fluo1, 2 for fluo2 etc..
        norm : boolean for normalizing or not
        '''
        #try:
        s = self.sum_fluo(c,i,col,norm=norm)
        if 1 in debug: print(f'self.sum_fluo is {self.sum_fluo}')
        # except:
        #     s = 0

        return s

    def mean_fluo(self,c,i,col,norm, debug=[]):
        '''
        Fluo normalized or not normalized divided by area..
        '''
        try:
            s = self.sum_fluo(c,i,col,norm=norm)/cv2.contourArea(c)
        except:
            s = 0

        return s

    def fluos(self, kind, col, norm=False, lim_fluo=[-1e6,1e8], debug=[]):
        '''
        make the list of the fluo integral for the cell during the time
        kind : sum or std
        col : color, an index : 1, 2 etc..
        norm : normalized
        '''
        self.obs = 'fluo'
        self.title = 'fluorescence evolution'
        self.xlabel = 'frames'
        self.ylabel = 'normalized fluorescence' if norm else 'fluorescence'
        list_fluos_ci = []
        if 0 in debug:
            print(f'len(self.list_cntrs_ci) is {len(self.list_cntrs_ci)} ')
            print(f'In fluos, kind is {kind}')
        for i,c in enumerate(self.list_cntrs_ci):                               # i is the image index
            # try:
            if kind == 'sum':
                sum = self.fluo_val(c,i,col,norm)
                if 1 in debug: print(f'integral fluo is {sum}')
                list_fluos_ci.append(sum)           # append the sum of fluo in function of the time
            elif kind =='std':
                list_fluos_ci.append(self.std_fluo(c,i,col))                # append the std of fluo in function of the time
            elif kind =='mean':
                list_fluos_ci.append(self.mean_fluo(c,i,col,norm))
            elif kind =='contrast':
                list_fluos_ci.append(self.contrast_fluo(c,i,col,norm))
            # except:
            #     print('Cannot make list for fluo..')

        cnd0 = max(list_fluos_ci) < lim_fluo[1]
        cnd1 = min(list_fluos_ci) > lim_fluo[0]

        if 0 in debug:
            print(f'#### cnd0 {cnd0}, cnd1 {cnd1}')
            print(f'max(list_fluos_ci) {max(list_fluos_ci)}')
            print(f'min(list_fluos_ci) {min(list_fluos_ci)}')
            print(f'list_fluos_ci {list_fluos_ci}')
        if kind != 'contrast':
            self.curr_obs = list_fluos_ci if cnd0 and cnd1 else np.zeros(len(list_fluos_ci))
        else:
            self.curr_obs = list_fluos_ci

        if self.fluo_bckgrd:
            # replace 0 by None
            self.curr_obs = [None if val==0 else val for val in self.curr_obs]
            self.curr_obs_mask = np.isfinite(np.array(self.curr_obs).astype(np.double))
            if 1 in debug:
                print(f'len(self.curr_obs) is {len(self.curr_obs)})')
                print(f'len(self.curr_obs_mask) is {len(self.curr_obs_mask)})')

        if 1 in debug:
            print(max(self.curr_obs))
            print(min(self.curr_obs))

        return self
