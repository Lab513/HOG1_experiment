# https://github.com/zhixuhao/unet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
op = os.path
opb = op.basename
opd = op.dirname


class HANDLE_FLY(object):
    '''
    Prepare images for training on the fly
    '''

    def __init__(self, *all_dir, kind='train',
                 format='tiff', dim=512,
                 batch_size=4,
                 image_color_mode="grayscale",
                 mask_color_mode="grayscale", image_save_prefix="image",
                 mask_save_prefix="mask", flag_multi_class=False,
                 num_class=2, save_to_dir=None,
                 seed=1):

        self.train_path, self.image_folder, self.mask_folder = all_dir
        print(f'### self.image_folder {self.image_folder}, \n'
              f'### self.mask_folder {self.mask_folder}, \n'
              f'### self.train_path {self.train_path}')
        self.format = format
        self.lenf = len(self.format)
        self.dim = (dim, dim)
        self.batch_size = batch_size
        self.flag_multi_class = flag_multi_class
        self.num_class = num_class
        self.addr_data = os.path.dirname(self.image_folder)

        self.aug_dict = dict(rotation_range=90,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             zoom_range=0.2)

        # featurewise_center=True,
        # featurewise_std_normalization=True,

        image_datagen = ImageDataGenerator(**self.aug_dict)
        mask_datagen = ImageDataGenerator(**self.aug_dict)
        self.image_generator = image_datagen.flow_from_directory(
            self.train_path,
            classes=[self.image_folder],
            class_mode=None,
            color_mode=image_color_mode,
            target_size=self.dim,
            batch_size=self.batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed)
        self.mask_generator = mask_datagen.flow_from_directory(
            self.train_path,
            classes=[self.mask_folder],
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=self.dim,
            batch_size=self.batch_size,
            save_to_dir=save_to_dir,
            save_prefix=mask_save_prefix,
            seed=seed)

    def train_generator(self, debug=[]):
        '''
        Generator of images and masks
        '''
        tg = zip(self.image_generator, self.mask_generator)

        for (img, mask) in tg:
            img, mask = self.adjustData(img, mask,
                                        self.flag_multi_class,
                                        self.num_class)
            if 0 in debug:
                print(f'### img.shape {img.shape},\n'
                      f'### mask.shape {mask.shape}')
            yield (img, mask)

    def adjustData(self, img, mask, flag_multi_class, num_class):
        '''
        '''
        if(flag_multi_class):
            img = img / 255
            mask = mask[:, :, :, 0]\
                if(len(mask.shape) == 4) else mask[:, :, 0]
            new_mask = np.zeros(mask.shape + (num_class,))
            for i in range(num_class):
                new_mask[mask == i, i] = 1
            new_mask = np.reshape(new_mask, (new_mask.shape[0],
                                  new_mask.shape[1]*new_mask.shape[2],
                                  new_mask.shape[3])) if flag_multi_class\
                                    else np.reshape(new_mask,
                            (new_mask.shape[0]*new_mask.shape[1],
                             new_mask.shape[2]))
            mask = new_mask

        elif(np.max(img) > 1):
            img = img / 255
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0

        return (img, mask)
