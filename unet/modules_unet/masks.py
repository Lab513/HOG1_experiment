from matplotlib import pyplot as plt
import cv2


class MASKS():
    '''
    Masks
    '''

    def make_mask_nuclei(self):
        '''
        '''
        self.img_mask_nuclei = self.img_mask.copy()
        # enhance the prediction mask
        self.img_mask_nuclei[self.img_mask > self.thresh_nuclei] = 255
        self.img_mask_nuclei[self.img_mask < self.thresh_nuclei] = 0

    def prepare_masks(self, debug=[0]):
        '''
        Thresholding, dilation and erosion
        '''
        if 0 in debug:
            print(f"##### in make_mask, self.thresh_after_pred"
                  f" is {self.thresh_after_pred}")
        if self.thresh_after_pred:
            # no thresholding if Stardist
            if self.model_alias[:2] != 'Sd':
                # thresh to 255
                self.img_mask[self.img_mask >= self.thresh_after_pred] = 255
                # thresh to 0
                self.img_mask[self.img_mask < self.thresh_after_pred] = 0
            try:
                self.img_mask_event[self.img_mask_event >
                                    self.thresh_after_pred] = 255
                self.img_mask_event[self.img_mask_event <
                                    self.thresh_after_pred] = 0
            except:
                print('cannot change mask for img_mask_event')

        if self.args.dilate_after_pred:
            print('### dilating predictions !!! ')
            # dilate the shapes in the mask before finding contours
            self.dilate_mask_shapes()
            self.morph_open(iter=10)

        if self.args.erode_after_pred:
            print('### eroding predictions !!! ')
            # erode the shapes in the mask before finding contours
            self.erode_mask_shapes(erd_size=self.erode_for_track,
                                   iter=self.iter_erode_for_track)
            #self.morph_open(iter=10)
