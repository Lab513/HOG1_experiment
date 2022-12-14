'''
Utilities..
If "save_data" in save_experim is set True the dataset is saved,
otherwise only the path is saved.
'''
import os
op = os.path
opb, opd, opj = op.basename, op.dirname, op.join
import shutil as sh
from colorama import Fore, Back, Style
from datetime import datetime
from time import time
#import argparse
from pathlib import Path
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

class UTIL(object):
    '''
    '''
    def __init__(self, test, model):
        self.model = model
        self.test = test
        #self.file = file

    def init_time(self):
        '''
        Trigger the chronometer
        '''
        self.t0 = time()

    def date(self):
        '''
        Return a string with day, month, year, Hour and Minute..
        '''
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M")
        return dt_string

    def show_time_calc(self):
        '''
        Show time elapsed since chronometer was triggered
        '''
        t1 = time()
        sec = round(((t1-self.t0)%60))
        min = (t1-self.t0)//60
        print('calculation time is {0} min {1} sec '.format(min,sec))

    def get_computing_infos(self):
        '''
        '''
        try:
            dl = device_lib.list_local_devices()[3]
            gpu_id = dl.physical_device_desc
            gpu_mem = str(round(int(dl.memory_limit)/1e9,2)) + ' MB'
            self.soft_hard_infos = {'id': gpu_id, 'mem': gpu_mem, 'tf_version': tf.__version__}
        except:
            print('issue with computing_infos')

    def save_computing_infos(self):
        '''
        Save computing informations about the training..
        '''
        self.get_computing_infos()
        with open(self.rep_save_exp / 'computing_infos.txt','w') as f:
            json.dump(self.soft_hard_infos, f)

    def inverted_models_name_dic(self, model):
        '''
        Invert mapping between long models name and models shortcuts
        '''
        addr_models_json = str(Path('modules_unet') / 'models.json')
        with open(addr_models_json, "r") as f:
            models = json.load(f)
        self.inverted_models = dict(map(reversed, models.items()))
        try:
            shortcut = self.inverted_models[model]
        except:
            shortcut = model  # in case shortcut does not exist..
        return shortcut

    def create_folder_for_pred(self, path_pred, debug=[0]):
        '''
        Name of the folder for the predictions
        '''
        short_model = self.inverted_models_name_dic(str(self.model))
        #suff = str(self.test) + '_' + self.file + '_' + self.date()
        #self.path_pred_model = path_pred / (short_model + '_test_' + suff)    # address of the prediction
        if short_model == 'ana_link':
            self.path_pred_model = path_pred / (short_model + '_' + self.date())
            self.path_pred_anaph = self.path_pred_model
            if 0 in debug:
                print(f'### self.path_pred_anaph is {self.path_pred_anaph}')
            os.mkdir(self.path_pred_model)
            self.path_lineage = opj(opd(path_pred), 'lineage')                            # folder for lineage
            os.mkdir(self.path_lineage)
        else:
            self.path_pred_model = path_pred / (short_model + '_track_' + self.date())       # address of the prediction
            os.mkdir(self.path_pred_model)                                                   # create the folder

    def make_predict_subdir(self, ha):
        '''
        Make sub directory for prediction
        '''
        self.ha = ha
        p = Path('./predictions')
        pred_name = self.test
        path_pred = p / pred_name                                               # path for the predictions
        print(f"## prediction path is {path_pred} ")
        if not os.path.exists(path_pred):
            os.mkdir(path_pred)                                               # folder movie for pedictions

        self.create_folder_for_pred(path_pred)

    def save_prediction(self, i, prediction, meth, debug=[]):
        '''
        Save the prediction
        '''
        if meth == 'unet':
            mask = prediction[0]*255
        elif meth == 'stardist':
            if 0 in debug: print(f'prediction[0].dtype is {prediction[0].dtype}')
            mask = (prediction[0]*65025).astype(np.uint16)
        addr_im_predicted = self.path_pred_model / (str(self.ha.tab_files[i]) + "png")
        print(addr_im_predicted)
        cv2.imwrite(str(addr_im_predicted), mask)
