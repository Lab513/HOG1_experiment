'''
https://matplotlib.org/stable/gallery/color/named_colors.html
'''
import os, re, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
op = os.path
opd, opb, opj, opa = op.dirname, op.basename, op.join, op.abspath
import json
import yaml
##
import shutil as sh
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
from skimage import io
from tensorflow.keras import models
from modules_analyse.BC import correctbaseline

class FIND_BAD_PICS():
    '''
    Detect bad pics and make a list
    '''
    def __init__(self, debug=[0]):
        #print(os.getcwd())
        self.colors = mcolors.CSS4_COLORS
        self.ind_bad_pics = []
        with open(opj('settings', 'bad_pics_model.yaml')) as f:                       # default address for saving the processings
            self.name_mod_segm = yaml.load(f, Loader=yaml.FullLoader)
        self.load_segm_model()
        if 0 in debug: print('Instantiate FIND_BAD_PICS ')

    def load_segm_model(self):
        '''
        load the model for the ML detection method
        '''
        addr_models_json = str(Path('modules_unet') / 'models.json')
        with open(addr_models_json, "r") as f:
            self.alias_models = json.load(f)
        self.model_segm = models.load_model(Path('models') / self.alias_models[self.name_mod_segm])

    def make_segmML_score(self, img, thresh=200, debug=[]):
        '''
        Calculate the score with ML segmentation
        '''
        if 0 in debug: print(f'In make_segmML_score, img.shape is {img.shape} !!! ')
        pred = self.model_segm.predict(img.astype('uint8'))[0]*255
        pred_thresh = pred.copy()
        pred_thresh[pred > thresh] = 255
        pred_thresh[pred < thresh] = 0
        pred_above_tresh = pred[pred > thresh]
        surf = pred_above_tresh.size
        self.quality_score = surf
        if 1 in debug: print(f' surf = {surf}')
        if 2 in debug:
            plt.imshow(pred_thresh)                                   # show prediction with threshold
            plt.show()

    def estimate_quality(self, addr_img, show_score=False):
        '''
        Estimate  the quality
        '''
        img = cv2.imread(addr_img)
        img = np.expand_dims(img, axis=0)
        self.make_segmML_score(img)
        if show_score: print(f'The score is {self.quality_score}')
        self.list_quality_scores.append(self.quality_score)

    def plot_quality(self, debug=[0]):
        '''
        '''
        if 0 in debug: print(f'In plot_quality !!!')
        fig = plt.figure()
        plt.title(f'find pics with pb')
        scores = self.list_quality_scores
        ##
        plt.ylim(0, max(scores)*1.3)
        ##
        plt.plot(scores)                                                       # plot curve of the scores
        plt.plot(self.lcorr)                                                   # Fitting curve unsing L1
        plt.plot(self.lcorr_tol, '--', color=self.colors['orange'])            # Low limit for error
        ##
        plt.xlabel('image index')
        plt.ylabel('surface of segmentation')
        ##
        for i,ind_bad in enumerate(self.ind_bad_pics):
            score_bad = self.y_bad_pics[i]
            plt.plot(ind_bad, score_bad, 'og')                                  # bad points, green points
        plt.savefig(str(Path(self.folder_analyzed) / f'img_quality.jpg'))
        plt.show(())
        plt.close(fig)

    def find_bad_pics(self, tol=0.85, lenchunk=7, nb_iter=2, debug=[0,1]):
        '''
        Find the data under the linear fit curve
        '''
        if 0 in debug: print(f'In find_bad_pics !!!')
        if 1 in debug:
            print('parameters are:')
            print(f'tol = {tol}')
            print(f'lenchunk = {lenchunk}')
            print(f'nb_iter = {nb_iter}')
        y = np.array(self.list_quality_scores)
        self.lcorr = -correctbaseline(-y, iterations=nb_iter,
                                      nbchunks=int(y.size/lenchunk))
        self.lcorr_tol = self.lcorr*tol
        self.ind_bad_pics = []
        self.y_bad_pics = []
        for i,c in enumerate(self.lcorr):
            if y[i] < tol*c:                               # under tolerance
                print(i)
                self.ind_bad_pics += [i]
                self.y_bad_pics += [y[i]]
        self.plot_quality()

    def take_num_frame(self, elem):
        '''
        frame34.tiff or frame34.png  --> 34
        '''
        return int(re.findall('\d+', elem)[0])

    def find_bad_pics_in_folder(self, folder, ext='png', debug=[]):
        '''
        ext: format of the test_images
        margin_down : margin for separating good pics for bad pics
        '''
        self.folder_analyzed = folder
        self.list_quality_scores = []                # list of scores
        if 0 in debug: print(f'folder with png is {folder}')
        ld = [opb(img) for img in glob.glob(f'{folder}/*.{ext}')]
        ld_sorted = sorted(ld, key = lambda i: self.take_num_frame(i)) #
        if 2 in debug: print(f'## ld_sorted  {ld_sorted}')
        for img in ld_sorted:
            self.estimate_quality(opj(folder, img))
        if 1 in debug:
            print(f'## self.list_quality_scores  {self.list_quality_scores}')
        # try:
        self.find_bad_pics()                   # find the bad pics
        # except:
        #     print('Cannot find bad pics')
