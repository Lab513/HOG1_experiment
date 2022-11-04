import os, sys, glob, re, json
op = os.path
opd, opb, opj = op.dirname, op.basename, op.join
opa = op.abspath
import yaml
import shutil as sh
from distutils.dir_util import copy_tree
from time import time
from datetime import datetime
import inspect as insp
from pathlib import Path
from PIL import Image
import numpy as np
from skimage import io
from numba import cuda
##
from matplotlib import pyplot as plt
##
import cv2
#
from utils.sep_tif_layers import SEP_TIF_LAYERS as STL
from analyse_results import ANALYSE_RESULTS as AR
from modules_unet.util_misc import date

def date():
    '''
    Return a string with day, month, year, Hour and Minute..
    Date used for local processed files..
    '''
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M")
    return dt_string

def timing(func, message=''):
    def wrapper(*args, **kwargs):
        t0 = time()
        func(*args, **kwargs)
        t1 = time()
        time_tot = round((t1-t0)/60,2)
        mess = 'for ' + message
        print(f'time elapsed {mess} is {time_tot} min')
    return wrapper

class FULL_PROC():
    '''
    Full processing ...
    '''
    def __init__(self, dir_exp, root_file, list_layers, final_dest, load_data=True, list_pos=None, temp_dest=None):
        '''
        dir_exp : folder of the experiment on the NAS
        root_file : name of the tif files without the index (2 last digits)
        list_layers : list of the layers used for the processings
        dest : final destination on the NAS
        load_data : if True load locally the dataset from the NAS
        list_pos : list of the tifs videos when loading in unet/temp folder ..
        temp_dest : temporary folder in case only the segm and track is needed and extraction yet done..
        '''
        self.list_pos = list_pos
        self.dir_exp = Path(dir_exp)                                  # address where are located the experimental datasets
        self.root_file = root_file                                    # name of the tif movie without 2 last digits
        self.root_file_dest = root_file.replace(' ','_')              # address for the processed datasets
        self.final_dest = final_dest
        self.make_local_dest_path(temp_dest)
        self.list_layers = list_layers
        self.retrieve_model_name()                                    # main model used for segmentation
        if load_data:
            self.load_data_in_temp()                                  # load data in temp directory before processsing
        else:
            self.dir_exp = Path(opj('temp', opb(self.dir_exp)))

    def find_tif_videos(self, addr_exp=None, debug=[0,1]):
        '''
        List of the tif videos in the experiment folder
        '''
        if self.list_pos:
            list_tifs = []
            for pos in self.list_pos:
                patt_tif = opj(addr_exp, f'*{str(pos).zfill(2)}*.tif')
                if 0 in debug:
                    print(f'patt_tif is { patt_tif }')
                tif = glob.glob(patt_tif)
                list_tifs += tif
            if 1 in debug:
                print(f'list_tifs is { list_tifs }')
        else:
            list_tifs = glob.glob(opj(addr_exp, f'*.tif'))

        return list_tifs

    def prepare_folders_temp(self):
        '''
        '''
        try:
            sh.rmtree('temp')
        except:
            pass
        self.temp_dir_exp = opj('temp', opb(self.dir_exp))             # temporary folder for local datasets
        try:
            os.mkdir('temp')
            os.mkdir(self.temp_dir_exp)
        except:
            pass

    @timing
    def load_data_in_temp(self):
        '''
        Load the data in local for processing faster
        '''
        self.prepare_folders_temp()
        print('###### Loading data to process to temp ######')
        list_tifs = self.find_tif_videos(addr_exp=self.dir_exp)
        for vid in list_tifs:
            sh.copy(vid, self.temp_dir_exp)                             # copy from distant to local, copy the whole experiment
            print(f'Loaded {opb(vid)}')
        print('Loaded all the data to process to temp !!!')
        self.dir_exp = Path(self.temp_dir_exp)                          # change the experimental dataset address form distant to local

    def make_local_dest_path(self, temp_dest, debug=[0]):
        '''
        Create the path self.temp_dest, temporary path for processings
        '''
        addr_proc = opj('settings', 'server_local_proc_addr.yaml' )
        with open(addr_proc, 'r') as f:
            root_dest = yaml.load(f, Loader=yaml.FullLoader)
            if temp_dest:
                self.temp_dest = opj(root_dest, temp_dest)
            else:
                self.temp_dest = opj(root_dest, f'proc_{date()}')
        if 0 in debug: print(f'Temporary destination folder for processings is {self.temp_dest} ')

    def retrieve_model_name(self, debug=[0]):
        '''
        Retrieve the name of the model
        '''
        addr_curr_mod = opj('settings', 'curr_model.yaml')          # model used for segmentation
        if 0 in debug:
            print(f'addr_curr_mod is {addr_curr_mod}')
        with open(addr_curr_mod, 'r') as f:
            self.name_mod = yaml.load(f, Loader=yaml.FullLoader)

    def num_file(self,i):
        '''
        Last two digits for position
        '''
        return f'0{i}' if i<10 else f'{i}'

    def merge_t0000_t0001_tifs(self, list_tif=range(25), root_file=None, suffix_file=None, debug=[0]):
        '''
        merge t000000 and t000001
        list_tif : list of the number identifying the tif files
        root_file : beginning of the name
        '''
        #
        tmerge0 = time()
        for i in list_tif:
            numf = self.num_file(i)
            f0 = f'{self.root_file}{numf}{suffix_file}0.tif'
            f1 = f'{self.root_file}{numf}{suffix_file}1.tif'
            addr0 = self.dir_exp / f0
            addr1 = self.dir_exp / f1
            print('Merging tifs...')
            if 0 in debug:
                print(f'addr0 is {addr0}')
                print(f'addr1 is {addr1}')
            stl = STL("a","b")
            stl.add_tif(str(addr0), str(addr1))                                   # Merge the tif files
            print('Merged...')
        tmerge1 = time()
        telapsed = round((tmerge1-tmerge0)/60, 2)
        print(f"time elapsed for merging is {telapsed} min")

    def convert_4layers_to_3layers_all_tifs(self, list_tif=None, root_file=None, suffix_file=None):
        '''

        '''
        #
        tconv0 = time()
        if not list_tif:
            list_tif = range(25)
        for i in list_tif:
            numf = self.num_file(i)
            f = f'{self.root_file}{numf}.tif'
            addr = opj(self.dir_exp, f)
            stl = STL(addr, self.temp_dest)
            stl.convert_4layers_to_3layers()
        tconv1 = time()
        print(f"In convert_4layers_to_3layers_all_tifs, time elapsed is {(tconv1-tconv0)/60} min")

    def clear_gpu(self):
        '''
        Clear the GPU memory
        '''
        device = cuda.get_current_device()
        device.reset()

    def copy_comp_tif_close_to_extract(self, addr0):
        '''
        Copy the composite tif file next to the extractions (BF, RFP etc..)
        '''
        name_tif = opb(addr0)
        name_pos_folder = name_tif.replace(' ', '_')[:-4]
        dest = opj(self.temp_dest, name_pos_folder, name_tif)
        print(f'Copy {addr0} to {dest}')
        sh.copy(addr0, dest)                                                                             # copy the composite tif file with the extraction

    def extract_layers_of_list_tif(self, list_tif=range(25), cpdest=True, merge=False):
        '''
        Extract the layers (BF and fluo layers)
        '''
        tt0_extract = time()
        for i in list_tif:
            t0_extract = time()                                                                # time for extract
            if merge:
                self.merge_t0000_t0001_tifs(list_tif=[i],\
                            root_file=self.root_file, suffix_file=merge)                       # Merge the tifs
            numf = self.num_file(i)
            f0 = f'{self.root_file}{numf}.tif'                                                 # Current composite tif file
            addr0 = opj(self.dir_exp, f0)                                                      # tif address in temp
            stl = STL(addr0, self.temp_dest)                                                   # separate tif layers
            stl.list_layers = self.list_layers                                                 #['BF','fluo1'] #,'fluo2'
            stl.extract_all_layers(rem_bad_pics=True, keep_ref_vid=True)
            if cpdest : stl.cp_to_dest()                                                       # copy the extractions to the temporary destination
            self.copy_comp_tif_close_to_extract(addr0)
            t1_extract = time()
            telapsed = round((t1_extract-t0_extract)/60,2)
            print(f"time elapsed for extracting for position {i} is {telapsed} min")           # time for one tif extraction
        tt1_extract = time()
        ttelapsed = round((tt1_extract-tt0_extract)/60,2)
        print(f"time elapsed for extracting all the pos is {ttelapsed} min")                   # time for all the extractions

    def find_clean_avi(self, pos, debug=[0]):
        '''
        Finding the address of the cleaned BF avi file for segmentation
        '''
        numf = self.num_file(pos)
        root_pattern = self.temp_dest + f'/{self.root_file_dest}{numf}/'
        search_pattern = f'{root_pattern}*cleaned_BF.avi'                     # search with cleaned_BF.avi
        if 0 in debug:
            print(f'numf is {numf} ')
            print(f'self.temp_dest is {self.temp_dest} ')
            print(f'search_pattern is {search_pattern} ')
        try:
            addr_clean = glob.glob(search_pattern)[0]
        except:
            search_pattern = f'{root_pattern}*_BF.avi'                        # if no existing cleaned_BF, search just for BF
            addr_clean = glob.glob(search_pattern)[0]
        if 0 in debug:
            print(f'addr_clean is {addr_clean} ')

        return addr_clean

    def copy_one_level_above(self, ll, dir_proc, debug=[0]):
        '''
        ll : list of files and folders
        dir_proc : path of the folder above the one with the processings
        '''
        for l in ll:
            if 0 in debug:
                print(f'file or folder is {l} ')
            try:
                sh.copytree(l, opj(dir_proc, opb(l)))
            except:
                pass
            try:
                sh.copy(l, dir_proc)
            except:
                pass

    def extract_proc_up(self,pos, debug=[0]):
        '''
        Extract the processing one level above..
        '''
        numf = self.num_file(pos)
        curr_fold = self.temp_dest + f'/{self.root_file_dest}{numf}'
        dir_proc = f'{curr_fold}/processings'
        if 0 in debug: print(f'dir_proc is {dir_proc}')
        ll = glob.glob(f'{dir_proc}/proc_*/*')
        name_fold = glob.glob(f'{dir_proc}/proc_*')[0]                       # folder with the procesings
        self.copy_one_level_above(ll, dir_proc)
        sh.rmtree(name_fold)                                                   # remove the folder

    def segm_track_list_tif(self, list_tif=range(25), name_mod=None, nb_imgs_proc=-1, debug=[1,2]):
        '''
        Segment and track cells
        nb_imgs_proc : number of images processed..
        '''
        if name_mod:
            self.name_mod = name_mod
        self.clear_gpu()                                                                                    # clear GPU before segmentation and tracking
        if 1 in debug:
            print(f'temporary destination, self.temp_dest is {self.temp_dest} ')
            print(f'final destination is {self.final_dest}')
        for pos in list_tif:                                                                                # segmentation and tracking for each tif
            addr_BF = self.find_clean_avi(pos)                                                              # using the cleaned avi for segmentation
            tproc0 = time()
            target = opj(opd(addr_BF), 'processings')  # , 'processings'                               # folder for processed datasets (cntrs, tracking movie etc..)
            args = f'--video --track all --num_cell --save_in {target}\
                             --kind_track min --nb_imgs_proc {nb_imgs_proc}'
            # args = f'--video '
            proc_args = f'-f {addr_BF} -m {self.name_mod} {args}'
            comm = f'python detect_cells.py {proc_args}'
            if 2 in debug:
                print(f'comm is {comm} ')
            print(f'os.getcwd() is {os.getcwd()}')
            os.system(comm)                                                                                 # execute the segmentation and tracking
            tproc1 = time()
            telapsed = round((tproc1-tproc0)/60,2)
            print(f"time elapsed for segm and track for pos {pos} is {telapsed} min")
            #self.extract_proc_up(pos)                                                                        # take results out of processings/proc_{date} and put in processings/
            try:
                sh.rmtree(opj('predictions', 'movie'))
            except:
                print('Cannot remove predictions folder')
        try:
            sh.copytree(self.temp_dest, self.final_dest)                                                       # Save to final destination, eg:on the NAS
        except:
            print('Cannot copy proc folder on the NAS..')

    def prepare_folders(self,num):
        '''
        Prepare folders for extraction and processings
        '''
        numf = self.num_file(num)
        addr = opj(self.temp_dest, f'{self.root_file_dest}{numf}')

    def make_sorted_list_cntrs(self):
        '''
        '''
        l0 = glob.glob(self.temp_dest + '/*/processings/proc_*-*/pkl_cntrs_[!pred]*')
        l0_sorted = sorted(l0, key= lambda elem : int(re.findall('f00(\\d+)', elem)[0]))
        return l0_sorted

    def make_addr(self, num, l0_sorted):
        '''
        '''
        numf = self.num_file(num)
        addr_results = l0_sorted[num]
        addr_fluo = opj(self.temp_dest, f'{self.root_file_dest}{numf}')
        addr_tif = opj(self.dir_exp, f'{self.root_file}{numf}.tif')
        print(f'addr_results is {addr_results}')
        print(f'addr_fluo is {addr_fluo}')
        return addr_results, addr_fluo, addr_tif
