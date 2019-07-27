# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cv2

from keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range = 180,
                               width_shift_range = 0.1,
                               height_shift_range = 0.1,
                               rescale= 1/255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True,
                               vertical_flip = True,
                               fill_mode = 'nearest'
                               )


from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense


model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units= 128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units= 1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

train_image_gen = image_gen.flow_from_directory('DATA/train',
                                                target_size=(150,150),
                                                batch_size=16,
                                                class_mode='binary')


test_image_gen = image_gen.flow_from_directory('DATA/test',
                                                target_size=(150,150),
                                                batch_size=16,
                                                class_mode='binary')

train_image_gen.class_indices

results = model.fit_generator(train_image_gen,epochs=200,
                              steps_per_epoch=150,
                              validation_data=test_image_gen,
                              validation_steps=12)

model.save('model.h5')

plt.plot(results.history['acc'])








