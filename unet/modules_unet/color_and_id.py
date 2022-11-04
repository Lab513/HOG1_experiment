import cv2
import numpy as np

class COLOR_AND_ID():
    '''
    Color and put the cell's ID
    '''

    def __init__(self):
        '''
        '''

    def pos_num(self,i):
        '''
        Position for the cell number in the picture
        '''
        shift = 5
        pos = np.array(self.list_pos[i])
        if pos[1] < 500:
            pos += np.array([-shift, shift])
        return tuple(pos)

    def insert_num_cell(self,i):
        '''
        number of the cell for tracking
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.3
        color = (255,)*3
        pos = self.pos_num(i)
        if self.args.num_cell:                # show the cell number
            cv2.putText(self.img_colid, str(i),\
                    pos, font, size, color, 1, cv2.LINE_AA)

    def debug_pos_i(self, pos, i):
        '''
        '''
        print(f'index is {i} ')
        print(f'pos is {pos} ')
        print(f'self.curr_contours {self.curr_contours[i]}')

    def fill_pred(self, i, mask, debug=[0]):
        '''
        Fill prediction mask
        '''
        if 0 in debug:
            print(f'# In fill_pred, i is {i}')
        self.img_colid[mask > 254]  = self.curr_rand_col[i]    # color by thresholding with mask

    def color_num_cell(self, i, mask):
        '''
        Give a color and an Id number to the cell
        '''
        self.fill_pred(i,mask)                            # fill the prediction mask with color
        self.insert_num_cell(i)                           # show the Id num of the cell

    def color_and_id_for_cells(self, img_colid, rand_col, debug=0):
        '''
        Identify the cells
        '''
        self.img_colid = img_colid
        self.curr_rand_col = rand_col
        list_pos = self.list_pos        # self.choose_list_pos()
        self.curr_mask = {}
        for i, pos in enumerate(list_pos):
            if debug > 0: self.debug_pos_i(pos, i)
            if self.args.erode_after_pred:
                dil, iter_dil = self.dil_last_shape, self.iter_dil_last_shape
            else:
                dil, iter_dil = 1,1
            mask = self.make_mask_from_contour(self.curr_contours[i], dilate = dil, iter_dilate = iter_dil)  # mask from contours with dilation
            if self.args.track != 'all':
                if i in self.args.track:
                    self.color_num_cell(i,mask)                                 # color the cell and number it
            else:
                self.color_num_cell(i,mask)
