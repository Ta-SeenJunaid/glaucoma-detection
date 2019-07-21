# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range = 180,
                               width_shift_range = 0.1,
                               height_shift_range = 0.1,
                               rescale= 1/max_val,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True,
                               vertical_flip = True,
                               fill_mode = 'nearest'
                               )