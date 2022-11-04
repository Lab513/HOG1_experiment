import cv2


class SINGLE_IMAGE():
    '''
    Analysis for all cells in an image
    '''
    def __init__(self):
        pass

    def image(self, i):
        '''
        Select an image
        '''
        self.img_id = i                             # identity of image
        self.list_cntrs_img_i = []
        for c in self.cntrs[i]:
            # list of the contours in the image
            self.list_cntrs_img_i.append(c)
        return self

    def area_hist(self, nbbins=10):
        '''
        histogram of cells surface in the image
        '''
        self.obs = 'area_hist'
        self.title = 'area histogram'
        # list of the cells areas in the image
        list_areas = self.make_list_areas(self.list_cntrs_img_i)
        self.curr_obs = list_areas
        self.nbbins = nbbins
        return self

    def cells(self):
        '''
        Select all cells
        For counting etc..
        '''
        self.all_cells = True              # take into account all the cells

        return self

    def count(self):
        '''
        Make the list of nb of cells
        '''
        self.nb_cells = []
        self.obs = 'count'
        for _, lc in self.cntrs.items():
            self.nb_cells.append(len(lc))
        self.curr_obs = self.nb_cells
        self.title = 'count cells'
        return self

    def mean_fluo_for_one_image(self, j, lc, col, debug=[]):
        '''
        Mean fluo for one image
        '''
        normed_sum_fluo = 0
        nbc = 0
        for c in lc:
            # try:
            area = cv2.contourArea(c)
            if area > 10:
                # normalized integral of fluo
                normed_sum_fluo += self.sum_fluo(c, j, col)/area
                nbc += 1
            # except:
            #     print('probably issue with null area')
        avg_over_img = normed_sum_fluo/nbc
        if 0 in debug:
            print(f'Mean fluo is {avg_over_img} ')
        # averaged over all the cells in the image
        self.list_normed_sum_fluo.append(avg_over_img)

    def mean_fluo_over_images(self, img_range=None, col='1', debug=[]):
        '''
        Mean fluo over images..
        img_range : range over which the mean is observed.
        '''
        self.list_normed_sum_fluo = []
        self.obs = 'image_fluo'
        self.sum_avg_over_img = 0
        mean_range = range(img_range[0], img_range[-1])
        for j, lc in self.cntrs.items():
            if img_range:
                if j in mean_range:
                    if 1 in debug:
                        print(f'Mean fluo for image {j}')
                    self.mean_fluo_for_one_image(j, lc, col)
            else:
                self.mean_fluo_for_one_image(j, lc, col)
        if img_range:
            self.fluo_bckgrd = round(sum(self.list_normed_sum_fluo)/len(mean_range), 1)
            print(f'self.fluo_bckgrd = {self.fluo_bckgrd}')
        self.curr_obs = self.list_normed_sum_fluo
        self.title = f'fluo average over frames'
        self.xlabel = 'frames'
        self.ylabel = 'normalized fluorescence'
        return self
