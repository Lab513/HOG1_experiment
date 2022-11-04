import cv2
import numpy as np
from scipy.linalg import norm

class TRACK_MISC():
    '''
    Tracking and Segmentation miscellaneous methods
    '''

    def __init__(self):
        '''
        '''
        pass


    def debug_add_neightbours(self,i,j):
        '''
        '''
        if j == 21 and i == 20:
            print("i ",i)
            print("j ",j)
            print('### sum_radii ', sum_radii)
            print('### distij ', distij)

    def add_neighbours(self,i,j,debug=0):
        '''
        add neighbours to self.dic_neighbours
        '''
        distij = self.dist_ij(i,j)
        radiusi, radiusj = self.radii_ij(i,j)
        sum_radii = radiusi + radiusj
        if debug > 0 : self.debug_add_neightbours(i,j)
        if sum_radii > distij:
            self.dic_neighbours[i] += [j]    # add neighbour to i
            self.dic_neighbours[j] += [i]    # add neighbour to j

    def draw_neighbours(self,i):
        '''
        Draw a circle on the neighbours of contour i and on contour i
        '''
        self.img = cv2.circle(self.img, self.list_prev_pos[i], 5, self.rand_col[i], -1)
        print('self.dic_neighbours ', self.dic_neighbours)
        print("### self.dic_neighbours[i] ", self.dic_neighbours[i])
        for j in self.dic_neighbours[i]:
            self.img = cv2.circle(self.img, self.list_prev_pos[j], 3, self.rand_col[i], 2)

    def find_neighbours(self):
        '''
        Find the neighbours for each cell using the masks contour
        '''
        self.dic_neighbours = {i:[] for i in range(len(self.list_pos))}
        for i, posi in enumerate(self.list_pos):
            for j in range(i+1, len(self.list_pos)):
                self.add_neighbours(i,j)

    def largest_contour(self, contours):
        '''
        Largest contour
        '''
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        return cnt

    def pos_ij(self,i,j):
        '''
        positions i and j in numpy array format
        '''
        # posi = np.array(self.list_prev_pos[i])
        # posj = np.array(self.list_prev_pos[j])
        posi = np.array(self.list_pos[i])
        posj = np.array(self.list_pos[j])
        return posi, posj

    def vec_ij(self,i,j):
        '''
        Vector from i to j
        '''
        posi, posj = self.pos_ij(i,j)
        vecij = posj-posi
        return vecij

    def norm_vec_ij(self,i,j):
        '''
        Vector from i to j
        '''
        posi, posj = self.pos_ij(i,j)
        vecij = posj-posi
        norm_vecij = vecij/norm(vecij)
        return norm_vecij

    def perp_vec_ij(self,i,j):
        '''
        Make the perpendicular unit vector
        '''
        # posi, posj = self.pos_ij(i,j)
        # vecij = posj-posi
        # norm_vecij = vecij/norm(vecij)
        norm_vecij = self.norm_vec_ij(i,j)
        perp_vecij = np.zeros(2)
        perp_vecij[0] = norm_vecij[1]
        perp_vecij[1] = -norm_vecij[0]
        return perp_vecij

    def barycenter_ij(self,i,j):
        '''
        Find the barycenter of the cells i an j according to their size
        '''
        radi, radj = self.radii_ij(i,j)
        posi, posj = self.pos_ij(i,j)
        baryc_ij = posi + radi/(radi+radj)*self.vec_ij(i,j)
        return baryc_ij

    def find_minmax_maxmean(self, new_img):
        '''
        '''
        new_img_vec = new_img.reshape(1, new_img.size)[0]                       # from 2D to vector
        new_img_vec = new_img_vec[new_img_vec > 0]                            # remove 0 values
        vec_sort = new_img_vec.argsort()                                        # indices of sorted
        val_min_max = new_img_vec[vec_sort[:300]].max()                       # max of the min values
        val_max_mean = new_img_vec[vec_sort[::-1][:20]].mean()                # mean of max values
        diff = val_max_mean - val_min_max                                       # difference between mean of max values and max of min values..
        return val_min_max, val_max_mean, diff

    def plt_img(self, img, im_size=512):
        '''

        '''
        ##
        dpi_val = 100
        fig = plt.figure(figsize=(im_size/dpi_val, im_size/dpi_val), dpi=dpi_val)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img, aspect='equal', cmap='gray')

    def track_mess0(self):
        '''
        message 0
        '''

        print(Fore.GREEN + str(self.list_pos))
        print(len(self.list_pos))
        print(Style.RESET_ALL)

    def track_mess1(self):
        '''
        message 1
        '''

        print('##### saving old positions #####')
        print(Fore.YELLOW + str(self.list_prev_pos))
        print(len(self.list_prev_pos))
        print(Style.RESET_ALL)


    def get_radius(self,i):
        '''
        Equivalent radius for contour i
        '''
        area = cv2.contourArea(self.dic_dilated_contours[i])
        radius = np.sqrt(area/np.pi)
        return radius

    def radii_ij(self,i,j):
        '''
        Return the equivalent radii of i and j
        '''
        radiusi, radiusj = self.get_radius(i), self.get_radius(j)
        return radiusi, radiusj

    def dist_ij(self,i,j):
        '''
        Return the distance between i and j
        '''
        #posi, posj = self.list_prev_pos[i], self.list_prev_pos[j]
        posi, posj = self.list_pos[i], self.list_pos[j]
        distij = norm(np.array(posj) - np.array(posi))
        return distij

    def show_new_positions(self):
        '''
        Show new positions for debug
        '''
        for i in range(len(self.list_pos)):
            if i not in self.list_indices_pos:
                self.img = cv2.circle(self.img, self.list_pos[i], 5, (255,255,0), 1)

    def test_out_track_algo(self, old_pos, new_pos, debug=[]):
        '''
        Show the distances between new and previous position for all the cells
        '''
        ldiffnorm = []
        for i in range(min(len(old_pos), len(new_pos))):
            ldiffnorm += [round(norm(np.array(old_pos[i])-np.array(new_pos[i])),1)]
        if 0 in debug: print(f'*****!!!!#### ldiffnorm {ldiffnorm}')

    def circle_at_pos(self, pos, radius=10):
        '''
        '''
        try:
            #self.img_mask = cv2.circle(self.img_mask, pos, radius, (255, 0, 0), 2) # draw a circle around the center..
            self.img = cv2.circle(self.img, pos, radius, (255, 0, 0), 2) # draw a circle around the center..
        except:
            pass
