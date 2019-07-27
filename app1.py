# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(filters=3, kernel_size=(11,11), input_shape=(256, 256, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=96, kernel_size=(5, 5), input_shape=(256, 256, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=192, kernel_size=(3, 3), input_shape=(256, 256, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=192, kernel_size=(3, 3), input_shape=(256, 256, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units= 128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units= 1,activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()



