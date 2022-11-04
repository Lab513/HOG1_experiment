import os
op = os.path
opb, opd, opj = op.basename, op.dirname, op.join

from util_server import find_platform, chose_server
from util_misc import *
import pickle as pkl
from pathlib import Path
from time import time, sleep
from flask_socketio import emit
from analyse_results import ANALYSE_RESULTS as AR
from track_with_btrack import APPLY_BTRACK as AB


platf = find_platform()
server = chose_server(platf)

class SAVE_RESULTS():
    '''
    Save the results of the processings
    '''

    def save_obj_pkl(self, obj, name):
        '''
        Save the contours for further analysis with pkl format
        Dictionary of frames inside which are
         the list of contours in numpy array..
        '''
        with open(name, 'wb') as f:
            pkl.dump(obj, f)

    def save_cntrs_btrack(self):
        '''
        Save cntrs from btrack
        '''
        print(f' self.dir_result is {self.dir_result}')
        # reformat btrack tracks for pkl
        dic_cells = self.from_tracks_to_contours_format()
        # address where to save the btrack cntrs
        pkl_bt_cntrs_dir = self.generic_result_name_dir('pkl_bt_cntrs')
        try:
            self.pkl_bt_cntrs_dir_temp =\
              self.generic_result_name_dir_temp('pkl_bt_cntrs')
        except:
            print('no temp dir')
        # save btrack cntrs under pkl format
        self.save_obj_pkl(dic_cells, f'{pkl_bt_cntrs_dir}.pkl')
        try:
            # save in temp
            self.save_obj_pkl(dic_cells, f'{self.pkl_bt_cntrs_dir_temp}.pkl')
        except:
            print('no temp dir')

    def save_lineage(self, debug=[0]):
        '''
        Save lineage in results
        '''
        dest_lineage = self.dir_result / 'lineage'
        if 0 in debug:
            print(f'self.ut.path_lineage is {self.ut.path_lineage}')
            print(f'dest_lineage is {dest_lineage}')
        # try:
        # copy lineage in the processings
        copy_dir(Path(self.ut.path_lineage), dest_lineage)
        # except:
        #     print('Cannot copy lineage.. ')

    def save_tracks(self):
        '''
        Save tracks from btrack
        '''
        plk_tracks_dir = self.generic_result_name_dir('pkl_tracks')

        try:
            self.plk_tracks_dir_temp = self.generic_result_name_dir_temp('pkl_tracks')
        except:
            print('no temp dir')
        self.save_obj_pkl(self.tracks, f'{pkl_tracks_dir}.pkl')
        try:
            self.save_obj_pkl(self.tracks, f'{self.plk_tracks_dir_temp}.pkl')
        except:
            print('no temp dir')

    def save_all_contours(self):
        '''
        Save both the contours from post prediction
         and prediction in pkl format
        The object is a dictionary of which entries
         are the index of frame and the values the list of contours
        '''
        plk_cntrs_dir = self.generic_result_name_dir('pkl_cntrs')
        plk_cntrs_pred_dir = self.generic_result_name_dir('pkl_cntrs_pred')
        try:
            self.plk_cntrs_dir_temp =\
                self.generic_result_name_dir_temp('pkl_cntrs')
            self.plk_cntrs_pred_dir_temp =\
                self.generic_result_name_dir_temp('pkl_cntrs_pred')
        except:
            print('no temp dir')
        self.save_obj_pkl(self.contours, f'{plk_cntrs_dir}.pkl')
        # pickle the contours of predictions
        self.save_obj_pkl(self.contours_pred, f'{plk_cntrs_pred_dir}.pkl')
        try:
            self.save_obj_pkl(self.contours,
                              f'{self.plk_cntrs_dir_temp}.pkl')
            # pickle the contours of predictions
            self.save_obj_pkl(self.contours_pred,
                              f'{self.plk_cntrs_pred_dir_temp}.pkl')
        except:
            print('no temp dir')

    def clean_curr_pic_folder(self):
        '''
        Clean the folder for following the processing
        '''
        for f in os.listdir(self.folder_curr_pic):  # remove the pics
            print("interf_predict_track / static / curr_pic", f)
            try:
                os.remove(str(self.folder_curr_pic / f))
            except:
                pass

    def copy_curr_pic_in_server(self):
        '''
        Copy the current processed pic in the server
        so that the user can access from the interface
        to real time control of processing quality
        '''
        curr_frame = f'frame{self.num}.png'
        # address in predictions folder
        addr_pic_pred = self.dir_movie_pred / curr_frame
        # address in server
        self.addr_pic_server = self.folder_curr_pic / curr_frame
        # relative address in the server
        self.addr_relative_pic_server = Path('static')\
            / 'curr_pic' / curr_frame
        self.clean_curr_pic_folder()
        sh.copy(addr_pic_pred, self.addr_pic_server)

    def emit_addr_curr_pic(self):
        '''
        path for the pic lastly processed saved in the server
        '''
        # self.copy_curr_pic_in_server()
        # sleep(self.sleep_time)
        # emit address of the folder with images processed..
        # emit('res_dir', {'mess': str(self.addr_relative_pic_server)})
        # server.sleep(self.ts_sleep)

        try:
            self.copy_curr_pic_in_server()
            sleep(self.sleep_time)
            # emit address of the folder with images processed..
            emit('res_dir', {'mess': str(self.addr_relative_pic_server)})
            server.sleep(self.ts_sleep)
        except:
            print('cannot save pic in the server')

    def emit_proc_result_address(self):
        '''
        Emit the address of the processed images
        '''

        # sleep(self.sleep_time)
        # res_proc_static_addr = Path('static/processings')\
        # / opb(self.dir_movie_pred)
        # emit address of the folder with images processed..
        # emit('server_res_dir', {'mess': str(res_proc_static_addr)})
        # server.sleep(self.ts_sleep)

        try:
            sleep(self.sleep_time)
            res_proc_static_addr = Path('static') \
                / 'processings' / opb(self.dir_movie_pred)
            # emit address of the folder with images processed..
            emit('server_res_dir', {'mess': str(res_proc_static_addr)})
            server.sleep(self.ts_sleep)
        except:
            print('cannot emit address addr of resulting procs')

    def copy_in_dir(self, dir):
        '''
        Copy the folder containing the processings
        '''
        #[copy_dir(d, dir) for d in [self.proc_folder, self.orig_folder]]
        copy_dir(self.proc_folder, dir)

    def copy_in_dir_result(self):
        '''
        Copy the results in a folder with date
        '''
        self.copy_in_dir(self.dir_result)

    def copy_in_dir_result_temp(self):
        '''
        Copy the results in a temporary folder
        '''
        self.copy_in_dir(self.dir_result_temp)

    def copy_folders(self, save_temp=False):
        '''
        After each processing, copy the folder
         in self.dir_result and self.dir_result_temp
        '''
        # folder above folder results
        self.proc_folder = self.dir_movie_pred.resolve().parent
        print("### self.proc_folder is ", self.proc_folder)
        # folder above folder with original images
        self.orig_folder = self.output_folder.resolve().parent
        # processings/proc_ + date
        self.copy_in_dir_result()

        if save_temp:
            try:
                # processings/proc_temp
                self.copy_in_dir_result_temp()
            except:
                print('no dir temp')
        # static/processings (server)
        copy_dir(self.proc_folder, self.path_procs_static)
        # emit the address of the folder with processed images
        self.emit_proc_result_address()

    def save_proc_results(self):
        '''
        After the processing is achieved,
         save the results in 'proc_date' folder..
        '''
        if self.args.track or self.args.show_pred:
            try:
                # save contours from btrack under pkl format
                self.save_cntrs_btrack()
            except:
                print('No not save btrack pkl')
            # save the contours with simple tracking and tracking on pred
            self.save_all_contours()
            # save the folders
            self.copy_folders()
        if self.args.lineage:
            # copy lineage n results folder..
            self.save_lineage()
