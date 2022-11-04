'''
Utilities..
If "save_data" in save_experim is set True the dataset is saved,
otherwise only the path is saved.
'''

from colorama import Fore, Style
from datetime import datetime
from time import time
from matplotlib import pyplot as plt

import shutil as sh
import argparse
from pathlib import Path
import json
import tensorflow as tf
from tensorflow.python.client import device_lib
import sys
import os
op = os.path
opb, opd, opj = op.basename, op.dirname, op.join


class UTIL(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='train with u-net')

        parser.add_argument('-d', '--data', type=str,
                            help='dataset eg : training-cell')
        parser.add_argument('-n', '--net', type=str,
                            help='net used for training')
        parser.add_argument('-o', '--output', type=str, help='destination')
        parser.add_argument('-k', '--kind_augm', type=str,
                            help='mem or fly', default='mem')
        parser.add_argument('-e', '--nb_ep', type=int,
                            help='nb epochs', default=5)
        parser.add_argument('-p', '--path', type=str,
                            help='path to training set',
                            default='training_sets')

        self.args = parser.parse_args()

    def init_time(self):
        '''
        Trigger the chronometer
        '''
        self.t0 = time()

    def init_log(self):
        '''
        Init the log file
        '''
        sys.stdout = Logger('.')     # trigger the log file savings

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
        sec = round(((t1-self.t0) % 60))
        min = (t1-self.t0)//60
        print(f'calculation time is {min} min {sec} sec ')

    def save_experim(self, ha, save_data=False):
        '''
        Save all the code for training with the resulting model and dataset
        '''
        sh.copytree('modules_unet', self.rep_save_exp / 'modules_unet')
        if save_data:
            name_data = os.path.basename(ha.addr_data)
            sh.copytree(ha.addr_data, self.rep_save_exp / name_data)
        else:
            with open(self.rep_save_exp / 'data_path.txt', 'w') as f:
                f.write(ha.addr_data)
        sh.copy('train_unet.py', self.rep_save_exp)
        sh.copy('log.dat', self.rep_save_exp)

    def get_computing_infos(self):
        '''
        Infos about GPU ..
        '''
        try:
            dl = device_lib.list_local_devices()[3]
            gpu_id = dl.physical_device_desc
            gpu_mem = str(round(int(dl.memory_limit)/1e9, 2)) + ' MB'
            self.soft_hard_infos = {'id': gpu_id,
                                    'mem': gpu_mem,
                                    'tf_version': tf.__version__}
        except:
            print('issue with computing_infos')

    def save_computing_infos(self):
        '''
        Save computing informations about the training..
        '''
        self.get_computing_infos()
        with open(self.rep_save_exp / 'computing_infos.txt', 'w') as f_w:
            json.dump(self.soft_hard_infos, f_w)

    def save_training_history(self, my_model):
        '''
        Save the training history
        Contains the loss, accuracy, validation loss, validation accuracy
        '''
        hist = my_model.history
        # print(hist.params)
        with open(self.rep_save_exp / 'training_history.json', 'w') as f_w:
            json.dump(hist.history, f_w)
        self.plot_train_hist(hist)

    def plot_loss(self, loss, val_loss):
        '''
        '''
        plt.figure()
        plt.title('Loss')
        #plt.ylim(0, 1)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(loss, label='train')
        if val_loss:
            plt.plot(val_loss, label='val')
        plt.legend()
        plt.savefig(self.rep_save_exp / 'loss.png')

    def plot_accuracy(self, accuracy, val_accuracy):
        '''
        '''
        plt.figure()
        plt.title('Accuracy')
        plt.ylim(0, 1)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.plot(accuracy, label='train')
        if val_accuracy:
            plt.plot(val_accuracy, label='val')
        plt.legend()
        plt.savefig(self.rep_save_exp / 'accuracy.png')

    def plot_train_hist(self, hist):
        '''
        '''
        loss = hist.history['loss']
        accuracy = hist.history['accuracy']
        try:
            val_loss = hist.history['val_loss']
        except:
            val_loss = None
            print('No validation loss')
        try:
            val_accuracy = hist.history['val_accuracy']
        except:
            val_accuracy = None
            print('No validation accuracy')
        self.plot_loss(loss, val_loss)
        self.plot_accuracy(accuracy, val_accuracy)

    def error_missing_option_data(self):
        '''
        Alert if missing training dataset..
        '''
        print(Style.BRIGHT)
        print(Fore.RED + '## Need a dataset address'
                         ' for training: "--data address" ..')
        print(Style.RESET_ALL)

    def make_model_name(self, dil, flood, dic_proc_name):
        '''
        Make the name of the model
        '''
        name_proc0 = '{name}-ep{epochs}-bs{batch_size}'
        if dil > 1:
            name_proc0 += '-dil{dilation}'
        if flood:
            name_proc0 += '-fl'
        name_proc1 = name_proc0 + '_date{date}'
        name_proc = name_proc1.format(**dic_proc_name)
        return name_proc

    def make_savings(self, ha, models, my_model, epochs,
                     batch_size, dil, flood):
        '''
        Save the model and the informations around the experiment.
        '''
        dic_proc_name = {'name': self.args.data,
                         'epochs': epochs,
                         'batch_size': batch_size,
                         'dilation': dil,
                         'date': self.date()}
        name_proc = self.make_model_name(dil, flood, dic_proc_name)
        print('name_proc ', name_proc)
        self.show_time_calc()
        self.dest = Path('models') / name_proc  # self.args.output
        print(f'saving at address : {self.dest} ')
        self.rep_save_exp = self.dest / 'experiment'
        models.save_model(my_model, self.dest)
        self.save_experim(ha)
        self.save_computing_infos()
        self.save_training_history(my_model)
        self.save_model_outside('Z:/@Analyses_utilities/temp_models')

    def save_model_outside(self, addr_targ):
        '''
        Copy the model to another place..
        '''
        try:
            sh.copytree( self.dest, opj(addr_targ, opb(self.dest)) )
        except:
            print(f'Cannot copy the model {self.dest} to {addr_targ}')

    def inverted_models_name_dic(self, model):
        '''
        Invert mapping between long models name and models shortcuts
        '''
        addr_models_json = str(Path('modules_unet')/'models.json')
        with open(addr_models_json, "r") as f:
            models = json.load(f)
        self.inverted_models = dict(map(reversed, models.items()))
        try:
            shortcut = self.inverted_models[model]
        except:
            shortcut = model  # in case shortcut does not exist..
        return shortcut


class Logger(object):
    '''
    Logger
    '''
    def __init__(self, folder):
        self.terminal = sys.stdout
        self.log = open(opj(folder, 'log.dat'), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
