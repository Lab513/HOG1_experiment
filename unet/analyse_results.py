from modules_unet.read_extract_fluo import READ_EXTRACT_FLUO as REF
from modules_unet.count_fluo import COUNT_FLUO as CF
from modules_unet.fit_growth import FIT_GROWTH as FG
from modules_unet.plots import PLOTS as PL
from modules_analyse.single_cell import SINGLE_CELL as SG
from modules_analyse.single_image import SINGLE_IMAGE as SI
from modules_analyse.fluo_nucleus import FLUO_NUCLEUS as FN
from modules_analyse.fluo import FLUO as FL
from modules_analyse.util_cyber import CYBER as CY
from modules_analyse.miscellaneous import MISC as MI
from modules_analyse.test_analyse import TEST_ANALYSE as TA
from modules_analyse.BC import correctbaseline

from datetime import datetime
from time import time
import pickle as pkl
import socket

import glob
import os
op = os.path
opb, opd, opj = op.basename, op.dirname, op.join
import yaml
import json
import shutil as sh


from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from skimage import io
from scipy.linalg import norm
##
from scipy.interpolate import interp2d
from scipy.signal import savgol_filter
##
from IPython.display import IFrame
from IPython.core.display import display


def timing(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        func(*args, **kwargs)
        t1 = time()
        time_tot = round((t1-t0)/60, 2)
        args[0].time_tot = time_tot  # time_tot on self
        print(f'time elapsed is {time_tot} min')
    return wrapper

class ANALYSE_RESULTS(REF,CF,FG,PL,SG,SI,FN,FL,MI,CY,TA):
    '''
    Analysis of the images after contours detection..
    '''

    def __init__(self, addr_results=None, addr_fluo=None,
                 load_fluo=None, nb_fluo=None, ext='tif', targ_pos=None,
                 extract_addr_fluo=None, addr_buds=None, kind_track='simple',
                 name_exp='exp0', unique_csv=True,
                 load_visu=True):
        '''
        addr_results : folder with preprocessed files
        addr_fluo :
        load_fluo : which fluo channels to load (type list)
        nb_fluo : number of fluos on the video
        ext : extension of the video
        targ_pos : index of the analysed position
        extract_addr_fluo
        addr_buds
        kind_track : 'btrack' or 'simple', indicates which tracking
                        tool is used for producing the pickle object "cntrs"
                        by default Btrack is used
        name_exp : needed for producing the csv
        size : size of the pictures in the video
        load_visu : if True load the folders BF,
                    fluos and tracking for visualisation
        '''
        ldep = [REF,CF,FG,PL,SG,SI,FN,FL,MI,CY,TA]
        [dep.__init__(self) for dep in ldep]    # init for all the inheritances

        ##
        self.addr_results = addr_results
        self.nb_fluo = nb_fluo
        self.addr_buds = addr_buds
        self.kind_track = kind_track      # select the pkl bt_pkl or simple pkl
        self.ext = ext
        self.targ_pos = targ_pos
        ##
        self.dic_json = {}
        self.addr_fluo = addr_fluo                        # address images fluo
        self.load_fluo = load_fluo       # load locally the fluo from imgs_fluo
        self.extract_addr_fluo = extract_addr_fluo                #
        # maximal step allowded before cutting the track
        self.lim_step_max = 10
        self.list_imgs_fluo1 = []                  # list of the fluo1 pictures
        self.list_imgs_fluo2 = []                  # list of the fluo2 pictures
        self.load_files_for_analyse()            # load the contours  and films
        self.cf = CF()                           # count fluo with OTSU
        self.name_exp = name_exp
        self.unique_csv = unique_csv
        self.name_exp_csv = f'{self.name_exp}.csv'
        self.fluo_bckgrd = None
        try:
            os.remove(self.name_exp_csv)
        except:
            pass
        if load_visu:
            self.prepare_data_for_visu()

    @property
    def date(self):
        '''
        Return a string with day, month, year, Hour and Minute..
        '''
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M")
        return dt_string

    def prepare_data_for_visu(self):
        '''
        '''
        # images in the server for visualisation..
        self.copy_all_folders_for_visu()
        # nb of pics for the visu interface
        self.count_nb_of_pics()

    def save_experim_title(self):
        '''
        Save the title of the experiment for the interface
        '''
        name_exp = opb(self.addr_results)
        print(os.getcwd())
        with open('../experim_title.yaml', 'w') as f:
            yaml.dump(name_exp, f)

    def load_time_axis(self, addr_time_axis=None, debug=[0]):
        '''
        Load the time axis of the experiment
        '''
        self.time_axis = None
        try:
            # time axis address
            addr_time_axis = glob.glob(opj(self.addr_results,
                                 f'time_axis_pos{self.targ_pos}.yaml'))[0]
            with open(addr_time_axis) as f:
                # load time axis
                self.time_axis = yaml.load(f, Loader=yaml.FullLoader)
            print('Loaded time axis !!')
            if 0 in debug:
                print(f'addr_time_axis is {addr_time_axis}')
            if 1 in debug:
                print(f'self.time_axis is {self.time_axis}')
        except:
            print('No time axis found..')

    def load_colors(self):
        '''
        Load the colors for analysis
        '''
        addr_col = opj(self.addr_results, 'processings', 'colors.pkl')
        try:
            with open(addr_col, 'rb') as f:
                self.colors = pkl.load(f)
        except:
            print('Cannot find colors')

    def find_tif_video(self, addr_tif=None, debug=[0]):
        '''
        Find automatically the composite video for analysis
        '''
        if 0 in debug:
            print(f'self.addr_results is {self.addr_results} !!!')
        # try:
        addr_tif_res = glob.glob(opj(self.addr_results, '*.tif'))[0]
        # if 0 in debug:
        print(f'addr_tif_res is {addr_tif_res}')
        # except:
        #     print(f'Cannot find tif file at this address')

        addr_tif = addr_tif_res

        return addr_tif

    def copy_all_folders_for_visu(self, debug=[0]):
        '''
        Copy the folders with images for visualisation
        BF, fluos, track, lineage..
        '''
        for folder in ['BF', 'fluo1', 'fluo2']:
            addr_orig_folder = opj(self.addr_results, folder)
            # Copy original images for BF & fluos
            self.copy_folder_for_visu(addr_orig_folder, folder)
        ##
        path_track = opj(self.addr_results, 'processings', '*_track_*')
        if 0 in debug:
            print(f'path_track is {path_track}')
        addr_track_folder = glob.glob(path_track)[0]
        if 0 in debug:
            print(f'addr_track_folder is {addr_track_folder}')
        # Copy tracking images
        self.copy_folder_for_visu(addr_track_folder, 'track')
        ##
        # Copy lineage images
        addr_lineage_folder = opj(self.addr_results, 'processings', 'lineage')
        self.copy_folder_for_visu(addr_lineage_folder, 'lineage')
        ###
        print('All the folders for visu are ready !!!')

    def copy_folder_for_visu(self, addr_orig_folder, folder, debug=[0]):
        '''
        Copy the images for the visu server to show the results
        '''
        print(f'### Dealing with folder {folder}')
        path_visu = opj(opd(opd(os.getcwd())), 'simple_visu',
                        'static', 'pictures')
        print(f'path_visu is {path_visu}')
        try:
            addr_folder = opj(path_visu, folder)      # folder with png images
            try:
                sh.rmtree(addr_folder)
            except:
                print(f'Cannot erase the folder {folder}')
            sh.copytree(addr_orig_folder, addr_folder)
            if 0 in debug:
                print(f'Copied {folder} in visu..')
        except:
            print('Cannot copy the folder in visu..')

    def count_nb_of_pics(self):
        '''
        Number of pictures for visualisation
        '''
        addr_BF_folder = opj(self.addr_results, 'BF')
        ll = glob.glob(opj(addr_BF_folder, 'frame*.png'))
        nb_pics = len(ll)
        path_visu = opj(opd(opd(os.getcwd())), 'simple_visu')
        addr_nb_pics = opj(path_visu, 'nb_pics.yaml')
        with open(addr_nb_pics, 'w') as f:
            yaml.dump(nb_pics, f)

    def shapes_loaded(self, debug=[]):
        '''
        '''
        print(f'self.list_imgs_BF.shape is {self.list_imgs_BF.shape} ')
        try:
            print('self.list_imgs_fluo1.shape'
                  f' is {self.list_imgs_fluo1.shape} ')
        except:
            print('Cannot show the shape of list_imgs_fluo1')
        try:
            print('self.list_imgs_fluo2.shape'
                  f' is {self.list_imgs_fluo2.shape} ')
        except:
            print('Cannot show the shape of list_imgs_fluo2')

    def retrieve_size_from_BF(self):
        '''
        '''
        self.size = self.list_imgs_BF.shape[1]
        print(f'In retrieve_size_from_BF, self.size is {self.size} ')

    def load_shape4(self, debug=[0, 1, 2]):
        '''
        Load BF and fluos
        '''
        if 0 in debug: print('using load_shape4')
        setattr(self, f'list_imgs_BF', self.film[:, 0, :, :])        # load BF
        for col in self.load_fluo:
            if 1 in debug:
                print(f'Loading col {col}')
            # load fluos (shifted by int(col))
            setattr(self, f'list_imgs_fluo{col}', self.film[:, int(col), :, :])
        if 2 in debug:
            self.shapes_loaded()

    def load_shape3(self, debug=[0, 1, 2]):
        '''
        Load BF and fluos
        '''
        step = self.nb_fluo+1
        if 0 in debug: print('using load_shape3')
        if 1 in debug: print(f'step is {step}')
        setattr(self, f'list_imgs_BF', self.film[::step])        # load BF
        for col in self.load_fluo:
            if 1 in debug:
                print(f'Loading col {col}')
            # load fluos (shifted by int(col))
            setattr(self, f'list_imgs_fluo{col}', self.film[int(col)::step])
        if 2 in debug:
            self.shapes_loaded()

    def analyse_load_layers(self, debug=[]):
        '''
        Load BF and fluos
        '''
        print(f'self.addr_tif {self.addr_tif}')
        self.film = io.imread(self.addr_tif)
        print(f'self.film.shape is {self.film.shape} ')
        if len(self.film.shape) == 3:                            # eg: exp87
            self.load_shape3()
        elif len(self.film.shape) == 4:                          # eg: exp82
            self.load_shape4()
        self.retrieve_size_from_BF()                      # retrieve image size

    def deal_with_bad_pics(self):
        '''
        Replace bad pic with previous good pic
        '''
        self.load_rem_pictures()
        # correction on the video on the layers
        self.change_bad_pic_with_pic_before()

    def load_rem_pictures(self):
        '''
        Load the list of the filtered bad pictures
        '''
        addr_rem_pic = f'{self.addr_results}/rem_pics.json'
        try:
            with open(addr_rem_pic, 'r') as f:
                self.rem_pics = json.load(f)
        except:
            print('Cannot find self.rem_pics, self.rem_pics set to [] ')
            self.rem_pics = []

    def change_bad_pic_with_pic_before(self, debug=[]):
        '''
        Change each bad picture by the previous one
        '''
        print('Replacing bad pictures with previous ones...')
        for i in self.rem_pics:
            for kind in ['BF', 'fluo1', 'fluo2']:
                list_imgs = getattr(self, f'list_imgs_{kind}')
                if 0 in debug:                                            # check length of lists
                    print(len(list_imgs))
                if 1 in debug:                                            # check images replaced
                    #  Replace picture at i with picture at i-1
                    plt.imshow(list_imgs[i-1])
                try:
                    list_imgs[i] = list_imgs[i-1]
                    if 2 in debug:
                        print(f'For {kind}, replaced picture'
                              f' {i} by picture {i-1}')
                except:
                    if 2 in debug:
                        print(f'Cannot change pics in the video for {kind}')

    def find_cntrs_addr(self, debug=[0]):
        '''
        Find the address of the pickle file for the contours after tracking..
        '''
        root_cntrs = f'{self.addr_results}/processings'
        if self.kind_track == 'btrack':
            pkl_format = 'pkl_bt_cntrs'
        elif self.kind_track == 'simple':
            pkl_format = 'pkl_cntrs'
        addr_pkl = root_cntrs + f'/{pkl_format}_*.pkl'
        if 0 in debug:
            print(f'addr_pkl {addr_pkl}')
            print(f'glob.glob(addr_pkl) {glob.glob(addr_pkl)}')

        # retrieve pickle address of the contours
        # find the name with format pkl_format

        rem_patterns = ['cntrs_pred']
        addr_cntrs = [addr  for addr in glob.glob(addr_pkl)\
                      for patt in rem_patterns if patt not in addr][0]
        if 0 in debug:
            print(f'root_cntrs {root_cntrs}')
            print(f'addr_cntrs {addr_cntrs}')

        return addr_cntrs

    def find_max_cntrs(self, lc):
        '''
        Determine the maximum of cells through all the frames
        '''
        nb_cells = len([i for i in lc if i is not None])                      # neglecting the Nones for the format with None values
        if nb_cells > self.max_nb_cells:
            self.max_nb_cells = nb_cells

    def analyse_load_cntrs(self, debug=[]):
        '''
        Create the object "self.cntrs" from the pickle file containing the contours
        Pass from 512 to the current size of the images..(2048 on Dr Who, 512 on MM etc..)
        '''
        fact = int(self.size / 512)                                            # conversion size factor
        if 1 in debug: print(f'fact in analyse_load_cntrs is {fact} ')
        addr_cntrs = self.find_cntrs_addr()
        self.max_nb_cells = 0
        with open(addr_cntrs, 'rb') as f:
            dic_cntrs = pkl.load(f)
            self.cntrs = {}
            for j, lc in dic_cntrs.items():                                       # j: image index, lc : list contours
                self.cntrs[j] = []
                for c in lc:                                                     # contour in list contours
                    try:
                        self.cntrs[j] += [fact*c]                              # passing from 512 to 2048
                    except:
                        pass                         # Probably a None contour
                self.find_max_cntrs(lc)                 # self.max_nb_cells

        if 1 in debug:
            print(f'self.cntrs[0] {self.cntrs[0]}')
        if 2 in debug:
            print(f'len(self.cntrs) is {len(self.cntrs)} ')
        if self.time_axis:
            self.time_axis = self.time_axis[:len(self.cntrs)]                        # redefine time_axis length
        else:
            print('self.time_axis does not exist..')

    def analyse_extract_fluo(self):
        '''
        Extract the fluo images for video
        '''
        for i, addr in enumerate(self.extract_addr_fluo):
            kind_fluo = str(i+1)
            self.extract_fluo(addr, kind_fluo)                                 # extract imgs fluo from video

    def load_files_for_analyse(self, debug=[1]):
        '''
        Load the dictionary of the list of contours for each image
        '''
        self.save_experim_title()                                                # save the title of the experiment
        self.load_colors()                                                       # retrieve the colors
        self.load_time_axis()
        self.addr_tif = self.find_tif_video()                                    # tif address of the composite file
        if self.extract_addr_fluo:
            self.analyse_extract_fluo()
        if self.load_fluo:
            self.analyse_load_layers()                                           # load BF and fluo
        self.analyse_load_cntrs()                                                # load the contours
        if self.addr_buds:
            with open(self.addr_buds, 'rb') as f:
                self.buds = pkl.load(f)                                          # load the buds contours dictionary
        self.deal_with_bad_pics()                                                # make the corrections image[i] = image[i-1]

    def pos(self, c):
        '''
        position from contour
        '''
        x, y, w, h = cv2.boundingRect(c)
        pos = (int(x + w/2), int(y + h/2))
        return pos

    def fill_list_pos_interm(self, lc):
        '''
        Make the list of positions
        '''
        list_pos_interm = []
        for c in lc:
            pos = self.pos(c)
            list_pos_interm.append(pos)
        return list_pos_interm

    def find_positions(self):
        '''
        Make list of list of positions self.list_pos for each image
        '''
        self.list_pos = []
        for _, lc in self.cntrs.items():
            self.list_pos.append(self.fill_list_pos_interm(lc))                # add a list of positions

    def mask_from_cntr(self, cntr, mask=None):
        '''
        Find mask from contour
        '''
        try :
            s = mask.shape
        except:
            mask = np.zeros((self.size, self.size))
        cv2.drawContours(mask, [cntr], -1, (255, 255, 255), -1)                 # save mask of the segmentation
        return mask

    def make_masks_from_segm(self, num_img, addr_results, debug=[2]):
        '''
        Using segmentation for making masks
        num_img : image index
        addr_results : address of pickle file with the contours
        '''
        mask = np.zeros((self.size, self.size))
        for c in self.cntrs[num_img]:
            mask = self.mask_from_cntr(c, mask)                                 # adding contour c to the mask
        ##
        maskcv = cv2.cvtColor(np.float32(mask), cv2.COLOR_GRAY2BGR)
        addr_img_mask = str(addr_results / f'frame{num_img}.tif')
        if 1 in debug:
            print("## addr_img_mask ", addr_img_mask)
        cv2.imwrite(addr_img_mask, maskcv)          # save the mask
        if 2 in debug:
            plt.imshow(maskcv)

    def try_settings(self, attr, debug=[]):
        '''
        '''
        try:
            getattr(plt, attr)(getattr(self,attr))
        except:
            if 0 in debug:
                print(f'No attribute  {attr}')

    def settings_plot(self, xlim=None, ylim=None):
        '''
        Settings for the plot
        '''
        plt.title(self.title)
        self.try_settings('xlabel')
        self.try_settings('ylabel')
        if xlim:
            plt.xlim(*xlim)
        if ylim:
            plt.ylim(*ylim)

    def savisky(self, p0=21, p1=2):
        '''
        Applying Savisky Golay filter to self.curr_obs..
        '''
        self.curr_obs = savgol_filter(self.curr_obs, p0, p1)

    def corrL1(self, iterations=10, nbchunks=20):
        '''
        Efficient algorithm for removing fitting while removing the outliers
        iterations : number of iterations of the algorithms,
                     the more the smoother
        nbchunks : number of small chunks in which the curve is divided
                   for performing local fitting..
        '''
        self.curr_obs = correctbaseline(np.array(self.curr_obs),
                                        iterations, nbchunks)

    def zero_after_drop(self, drop=1e2, debug=[]):
        '''
        Set all the point to 0 after the detection of  big drop.
        '''
        curve = np.array(self.curr_obs)
        ldrops = np.where(np.diff(curve) < -drop)[0]
        if 0 in debug:
            print(f'ldrops is {ldrops}')
        try:
            self.curr_obs[ldrops[0]:] = 0
        except:
            pass

    def cure_plot(self):
        '''
        '''
        for i, v in enumerate(self.curr_obs):
            if i > self.cure_lim[1]:
                if v == 0 :
                    self.curr_obs[i] = self.curr_obs[i-1]
                # if (self.curr_obs[i]-self.curr_obs[i-1]) > 1e2:
                #     self.curr_obs[i] = self.curr_obs[i-1]
            elif i < self.cure_lim[0]:
                if v == 0:
                    self.curr_obs[i] = 100
                if (self.curr_obs[i]-self.curr_obs[i-1]) > 1e2:
                    self.curr_obs[i] = self.curr_obs[i-1]

    def plot(self, xlim=None, ylim=None, new_fig=False, xaxis=[],
             cure=False, cure_lim=[20, 100], link=[],
             title='', rounding=1, debug=[]):
        '''
        plot the current observation for the cell
        Can be : the position, fluorescence, area
        '''
        if 0 in debug:
            print(f'Current observation is  {self.obs}')
            print(f'in plot, len(self.curr_obs) is  {len(self.curr_obs)}')
        if self.obs not in ['nucl_fluo']:
            self.curr_obs = [round(pt, rounding)
                             if pt != None else pt for pt in self.curr_obs]
        if self.time_axis:
            # using time_axis from the experiment
            xaxis = self.time_axis
            print('Using time_axis !!!')
            if 1 in debug: print(f'self.time_axis is {self.time_axis}')
        if title != '':
            self.title = title
        if len(xaxis) == 0:
            # default axis
            xaxis = np.arange(len(self.curr_obs))
        else:
            if 1 in debug:
                print(f'using axis {xaxis}')
        try:
            if 0 in debug:
                print(f'len(xaxis) is {len(xaxis)}')
                print(f'in plot again, len(self.curr_obs)'
                      f' is {len(self.curr_obs)}')
                print(f'in plot, len(self.curr_obs_mask)'
                      f' is {len(self.curr_obs_mask)}')
            # Do not plot the None values
            self.curr_obs_xy = [xaxis[self.curr_obs_mask],
                                np.array(self.curr_obs)[self.curr_obs_mask]]
        except:
            if 1 in debug:
                print('not using mask...')
            self.curr_obs_xy = [xaxis, np.array(self.curr_obs)]  # No mask used
        for lim in link:
            print(f'lim[0],lim[1] is {lim[0],lim[1]}')
            print(f'lim[0] is {lim[0]}')
            if lim[0] < lim[1]:
                # points between limits have all value at lim[0]
                self.curr_obs_xy[lim[0]:lim[1]] = self.curr_obs_xy[lim[0]]
            else:
                if lim[1] == 0:
                    lim[1] = None
                    print('changing in None')
                self.curr_obs_xy[lim[0]:lim[1]:-1] = self.curr_obs_xy[lim[0]]
        if cure:
            self.cure_lim = cure_lim
            #self.cure_plot()
            self.savisky()
        # possibly BGR instead of RBG hence ::-1
        col = tuple(t/255 for t in self.colors[self.id][::-1])
        if new_fig:
            plt.figure()
        self.settings_plot(xlim=xlim, ylim=ylim)
        if self.obs == "track":
            # cells positions
            plt.plot(*self.curr_obs_xy, label=str(self.id), color=col)
        elif self.obs == "count":
            # graph of cell number
            plt.plot(*self.curr_obs_xy)
        elif self.obs == "area_hist":
            # histogram cells area
            plt.hist(*self.curr_obs_xy, self.nbbins)
        elif self.obs == "image_fluo":
            # normalized fluo in the time through images
            plt.plot(*self.curr_obs_xy, label=str(self.id), color=col)
        elif self.obs == "nucl_fluo":
            # coloc fluo in the time through images
            plt.plot(*self.curr_obs_xy, label=str(self.id), color=col)
        else:
            if not ylim:
                plt.ylim(0, max(self.curr_obs_xy[1])*1.25)
            # general plot
            plt.plot(*self.curr_obs_xy, label=str(self.id), color=col)
        plt.legend()
        return self

    def find_maxi(self, maxi, debug=[0]):
        '''
        Find the maximum in the curve
        '''
        # print(f'self.curr_obs) {self.curr_obs}')
        curr_maxi = max([i for i in self.curr_obs if i is not None])
        if curr_maxi > maxi:
            maxi = curr_maxi

        return maxi

    def select_all_cells(self):
        '''
        '''
        return range(self.max_nb_cells)

    def save_fluo_results(self,
                          zero_after_drop=False,
                          corrL1=False,
                          csv=False,
                          debug=[]):
        '''
        '''
        if corrL1:
            nbchunks = int(len(self.curr_obs)/2)
            self.corrL1(iterations=1, nbchunks=nbchunks)
        if zero_after_drop:
            self.zero_after_drop(drop=zero_after_drop)
        # try:
        self.plot()                                # make the plots
        try:
            self.maxi = self.find_maxi(self.maxi)                # maxi y for plots
        except:
            print('Cannot calculate self.maxi ..')
        if 0 in debug:
            print(f'csv has the value {csv}')
        if csv or self.make_csv:
            self.csv()                     # save the curves in csv file

    def fluo_analysis(self,
                      kind=None,
                      cells='all',
                      zero_after_drop=False,
                      corrL1=False,
                      csv=False,
                      debug=[0]):
        '''
        Fluo analysis for many cells (by default all)
        '''
        if 0 in debug:
            print('In fluo_analysis !!!')
        self.maxi = 0
        if cells == 'all':
            cells = self.select_all_cells()

        for indexc, c in enumerate(cells):
            # try:
            if 1 in debug:
                print(f'In fluo analysis, dealing with cell {c}')
            print(f'cell : {indexc}/{len(cells)}')
            if kind == 'whole_cell':
                self.cell(c).fluos('sum', self.fluo_num, norm=True)
                self.save_fluo_results(zero_after_drop=zero_after_drop,
                                       corrL1=corrL1,
                                       csv=csv)
            elif kind == 'nucleus':
                # try:
                self.cell(c).fluo_in_nucleus()
                self.save_fluo_results(zero_after_drop=zero_after_drop,
                                       corrL1=corrL1,
                                       csv=csv)
            elif kind == 'only_nucleus':
                # try:
                print('Only the mean fluo in the nucleus...')
                self.cell(c).fluo_in_nucleus('mean_nucl')
                self.save_fluo_results(zero_after_drop=zero_after_drop,
                                       corrL1=corrL1,
                                       csv=csv)
                # except:
                #     print('*** Cannot find colocalisation fluorescence.. ***')

            # except:
            #     print(f'Cannot produce curve for cell {c}')
        print(f'self.maxi is {self.maxi}')
        plt.ylim(0, self.maxi*1.5)

    @timing
    def all_fluo(self,
                 cells='all',
                 zero_after_drop=False,
                 corrL1=False,
                 csv=False):
        '''
        Plot the cells fluorescence..
        '''

        self.fluo_analysis(kind='whole_cell',
                           cells=cells,
                           zero_after_drop=zero_after_drop,
                           corrL1=corrL1,
                           csv=csv)

    @timing
    def all_fluo_in_nucleus(self,
                            cells='all',
                            zero_after_drop=False,
                            corrL1=False,
                            csv=False):
        '''
        Plot the cells fluorescence colocalization inside the nucleus..
        '''

        self.fluo_analysis(kind='nucleus',
                           cells=cells,
                           zero_after_drop=zero_after_drop,
                           corrL1=corrL1,
                           csv=csv)

    @timing
    def all_fluo_only_nucleus(self,
                            cells='all',
                            zero_after_drop=False,
                            corrL1=False,
                            csv=False):
        '''
        Plot the cells mean fluo
        '''

        self.fluo_analysis(kind='only_nucleus',
                           cells=cells,
                           zero_after_drop=zero_after_drop,
                           corrL1=corrL1,
                           csv=csv)

    def make_sum_curve(self, i, debug=[]):
        '''
        '''
        curve = np.array([i if i is not None else
                          self.fluo_bckgrd for i in self.curr_obs])
        self.curves += [curve]
        self.maxi_sum = self.find_maxi(self.maxi_sum)
        if 0 in debug:
            print(f'current index in sum_curve is {i}')
        if i == 0:
            self.sum_curve = curve.copy()
        else:
            self.sum_curve += curve
        self.nb_curves += 1

    def average_fluo(self, cells,
                     zero_after_drop=False,
                     corrL1=False,
                     csv=False):
        '''
        Plot the average fluorescence of the cells and the envelope.
        '''
        fig, ax = plt.subplots()
        if self.nb_curves > 0:
            av_curve = self.sum_curve/self.nb_curves
            std_slc = []
            for i in range(av_curve.size):
                sl = []
                for cv in self.curves:
                    sl.append(cv[i])
                std_slc += [np.array(sl).std()]
            std_slc = np.array(std_slc)
            plt.ylim(0,self.maxi_sum*1.5)
            ax.plot(av_curve)
            ax.fill_between(np.arange(len(av_curve)),
                            av_curve-std_slc, av_curve + std_slc, alpha=0.2)
        else:
            print('No curve for making the average..')

    def csv(self, debug=[]):
        '''
        Save self.curr_obs as csv file
        '''

        if 0 in debug:
            print('in csv !!')
            print(f'len(self.curr_obs) is {len(self.curr_obs)}')
        if len(self.curr_obs) == 2:
            # x, y components
            series = pd.Series(self.curr_obs[1])
        else:
            # y component
            series = pd.Series(self.curr_obs)
        addr_csv = f'{self.addr_results}/{self.experim_name}.csv'
        if self.unique_csv:
            try:
                dic_cell = {'cell' + str(self.id): series}
                # error if it does not exist
                df = pd.read_csv(addr_csv)
                new_col = pd.DataFrame(dic_cell)
                df = df.merge(new_col, left_index=True, right_index=True)
            except:
                # initialize dic_cell
                dic_cell = {}
                if self.time_axis:
                    # adding time to csv
                    dic_cell['time'] = self.time_axis
                if 'fluo' in self.experim_name:
                    if 1 in debug:
                        print(f'self.experim_name {self.experim_name}')
                        print('insert fluo_bckgrd in csv !!!')
                    dic_cell['fluo_bckgrd'] = pd.Series([self.fluo_bckgrd])
                dic_cell['cell' + str(self.id)] = series
                df = pd.DataFrame(dic_cell)
            df.to_csv(addr_csv, index=False)
        else:
            df = pd.DataFrame(series)
            df.to_csv(f'{self.addr_results}/'
                      f'{self.experim_name}_cell{self.id}.csv',
                      index=False, mode='w')

    def axis_scale(self, coord, offset=0, debug=[]):
        '''
        '''
        new_coord = (int(coord)-offset)/500*self.size

        return new_coord

    def retrieve_clicked_pos(self, f, debug=[]):
        '''
        retrieve frames and positions
        '''
        l_frm_pos = yaml.load(f, Loader=yaml.FullLoader)
        l_curr_frm_pos = []
        for frm_pos in l_frm_pos:
            if 0 in debug:
                print(f'pos is {frm_pos}')
            if frm_pos['x'] != 'all':
                x = self.axis_scale(frm_pos['x'], offset=50)
                y = self.axis_scale(frm_pos['y'], offset=50)
                curr_pos = np.array([x, y])
            else:
                curr_pos = None
            curr_frame = int(frm_pos['frame'])
            l_curr_frm_pos += [[curr_frame, curr_pos]]

        return l_curr_frm_pos

    def rect_pos(self, c):
        '''
        Return the position using boundingRect
        '''
        (x, y, w, h) = cv2.boundingRect(c)
        cx, cy = x+w/2, y+h/2
        pos = np.array([cx, cy])

        return pos

    def nearest_cntr_of_currpos(self, cntrs, curr_pos, debug=[]):
        '''
        Find the index of the nearest contour
        '''
        lpos = []
        for c in cntrs:
            try:
                lpos += [self.rect_pos(c)]
            except:
                pass
        # all distances from position pos
        ldist = list(map(norm, (lpos-curr_pos)))
        ind = np.argmin(ldist)

        return ind

    def retrieve_x1x2y1y2(self, frm_area, debug=[]):
        '''
        Area for selecting cells..
        '''
        x1 = self.axis_scale(min(frm_area['x1'], frm_area['x2']))
        x2 = self.axis_scale(max(frm_area['x1'], frm_area['x2']))
        y1 = self.axis_scale(min(frm_area['y1'], frm_area['y2']))
        y2 = self.axis_scale(max(frm_area['y1'], frm_area['y2']))
        if 0 in debug:
            print(f'x1,x2,y1,y2 {x1,x2,y1,y2}')

        return x1, x2, y1, y2

    def find_cells_index_area(self, areas, debug=[0, 1]):
        '''
        From areas in the window and the frame,
        find the indices of the cells..
        '''
        l_indices = []
        # list of cells indices
        l_frm_areas = yaml.load(areas, Loader=yaml.FullLoader)
        if 0 in debug:
            print(f'l_frm_areas {l_frm_areas}')
        for frm_area in l_frm_areas:             # retrieve the frame and [x,y]
            curr_frame = frm_area['frame']
            # retrieve a given area
            x1, x2, y1, y2 = self.retrieve_x1x2y1y2(frm_area)
            cntrs = self.cntrs[int(curr_frame)]
            for ind, c in enumerate(cntrs):
                pos = self.rect_pos(c)
                cndx = x1 < pos[0] < x2            # inside x interval
                cndy = y1 < pos[1] < y2            # inside y interval
                if cndx and cndy:
                    l_indices += [ind]             # inside --> save the index

        return l_indices

    def find_cells_index_pos(self, positions, debug=[]):
        '''
        From positions in the window and the frame,
        find the indices of the cells..
        '''
        # retrieve the selected positions or all
        l_curr_frm_pos = self.retrieve_clicked_pos(positions)
        l_indices = []
        if 0 in debug:
            print(f'### l_curr_frm_pos {l_curr_frm_pos}')
        if type(l_curr_frm_pos[0][1]) == np.ndarray:
            # retrieve the frame and [x,y]
            for curr_frame, curr_pos in l_curr_frm_pos:
                cntrs = self.cntrs[curr_frame]
                ind = self.nearest_cntr_of_currpos(cntrs, curr_pos)
                # indices of the selected contours
                l_indices += [ind]
                if 1 in debug:
                    print(f'curr_pos is {curr_pos}')
                    print(f'ind is {ind}')
        else:
            curr_frame, _ = l_curr_frm_pos[0]     # retrieve the frame
            cntrs = self.cntrs[curr_frame]
            for ind, c in enumerate(cntrs):
                if type(c) == np.ndarray:         # if c != None
                    l_indices += [ind]            # indices of all the contours
        if 2 in debug:
            print(f'l_indices is {l_indices}')
        return l_indices

    def growth_curve(self):
        '''
        Number of cells with time
        '''
        growth = []
        for _, frame_cntrs in self.cntrs.items():
            growth += [len(frame_cntrs)]            # list with nb of contours
        plt.title('growth curve')
        plt.ylabel('#cells')
        if self.time_axis:
            plt.xlabel('time')
            plt.plot(self.time_axis, growth)        # use time axis
        else:
            plt.xlabel('frames')
            plt.plot(growth)

    def keep_if_no_zero(self, ind):
        '''
        '''
        if not 0 in self.curr_obs:
            self.plot(title=f'whole cell fluo evolution')     # fluo whole cell
            if self.make_csv:
                self.csv()
        else:
            print(f'Remove cell {ind}')

    def make_curves(self, indices, kind, debug=[]):
        '''
        indices : indices of the cells
        kind : which quantity we want to observe, area,
               fluo whole cell, average fluo, colocalization
               if "after fluo whole cell no_norm" --> no normalization
               by default it is normalized
        '''
        step_index = 5
        print(f'print every {step_index} ')
        self.kind_visu = kind
        suff = f'pos{self.targ_pos}_{self.kind_visu}'
        self.experim_name = f'{self.name_exp}_{suff}'
        cnd0 = kind in ['fluo whole cell', 'area']      # conditions for ylim
        ylim = 0
        if self.make_avg:
            self.nb_curves = 0
            self.curves = []
            self.maxi_sum = 0
        for i, ind in enumerate(indices):
            if 0 in debug:
                print(f'fluorescence evolution for index {ind}')
            if i%step_index == 0:
                print(f'Dealing with {i}/{len(indices)}')
            if 'fluo whole cell' in kind:
                if 'no_norm' in kind:
                    norm = False
                else:
                    norm = True

                # fluorescence

                self.cell(ind).fluos('sum', self.fluo_num, norm=norm)
                if self.rem_if_zero:
                    self.keep_if_no_zero(ind)
                else:
                    # fluo whole cell
                    self.plot(title=f'whole cell fluo evolution')
                    if self.make_csv:
                        self.csv()

                # Curves average

                if self.make_avg:
                    if self.rem_if_zero:
                        if not 0 in self.curr_obs:
                            self.make_sum_curve(i)
                    else:
                        self.make_sum_curve(i)

            elif kind == 'area':
                # area
                self.cell(ind).areas().plot(title=f'surface evolution')
                if self.make_csv:
                    self.csv()
            if cnd0:
                ylim = max(ylim, max([pt for pt in
                                      self.curr_obs if pt != None]))
        if kind == 'average fluo':
            self.average_fluo(indices,                        # average fluo
                              self.make_zero_after_drop,
                              self.make_corrL1,
                              self.make_csv)
        elif kind == 'colocalization':
            self.all_fluo_in_nucleus(indices,               # colocalization
                                     self.make_zero_after_drop,
                                     self.make_corrL1,
                                     self.make_csv)
        elif kind == 'only_nucleus':
            self.all_fluo_only_nucleus(indices,               # colocalization
                                     self.make_zero_after_drop,
                                     self.make_corrL1,
                                     self.make_csv)
        elif kind == 'growth':
            self.growth_curve()
        if cnd0:
            if 1 in debug:
                print(f'ylim is {ylim}')
            plt.ylim(0, ylim*1.3)               # max y coord
        if self.make_csv:
            print(f'Saved in csv at address'
                  f' {self.addr_results}/{self.experim_name}.csv')
        if self.save_fig:
            addr_img = f'{self.addr_results}/{self.experim_name}.png'
            plt.savefig(addr_img)
            print(f'Saved image at address {addr_img}')

    def visu(self, kind,
             fluo_num=1,
             make_avg=False,
             zero_after_drop=False,
             corrL1=False,
             csv=False,
             save_fig=False,
             rem_if_zero=False,
             debug=[1]):
        '''
        show specific analysis
        area evolution, fluo whole cell, fluo average, colocalization..
        kind : which quantity we want to observe, area,
               fluo whole cell, average fluo, colocalization
        zero_after_drop : Boolean, when the curve drops
                          set the next points to 0
        corrL1 : Boolean, L1 correction for a smooth curve
        csv : Boolean for producing csv
        save_fig : save the figure
        rem_if_zero : remove the curve if one or various zeros inside
            cannot be used in cells die since it removes curves we want
            to conserve.
        '''
        if 0 in debug:
            print(f'csv value is {csv}')
        self.make_zero_after_drop = zero_after_drop
        self.make_corrL1 = corrL1
        self.make_csv = csv
        self.save_fig = save_fig
        self.rem_if_zero = rem_if_zero
        self.fluo_num = fluo_num
        self.make_avg = make_avg
        indices_pos = indices_area = []
        try:
            with open('../pos.yaml', 'r') as positions:
                # cells indices for positions
                indices_pos = self.find_cells_index_pos(positions)
        except:
            pass
        #try:
        # areas coordinates
        with open('../selected_area.yaml', 'r') as areas:
            # cells indices for areas
            indices_area = self.find_cells_index_area(areas)
        # except:
        #     pass
        indices = indices_pos + indices_area
        if 1 in debug:
            print(f'indices {indices}')
        self.make_curves(indices, kind)

    def get_ip_address(self):
        '''
        IP address for local usage
        '''
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

    def display(self, width=1200, height=700,
                addr='local', port=5975):
        '''
        Display the films in BF, fluo, tracking..
        width : width of the image
        height : height of the image
        addr : local or server
        '''
        if addr == 'local':
            # local ip address
            ip_addr = f'http://{self.get_ip_address()}'
        elif addr == 'server':
            # server ip address
            ip_addr = 'http://desktop-4p4309t'
        # display the interface in Jupyter Notebook
        display(IFrame(f'{ip_addr}:{port}', str(width), str(height)))
