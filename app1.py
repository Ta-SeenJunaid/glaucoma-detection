# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(filters=3, kernel_size=(11,11), input_shape=(150, 150, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=96, kernel_size=(5, 5), input_shape=(150, 150, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=192, kernel_size=(3, 3), input_shape=(150, 150, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=192, kernel_size=(3, 3), input_shape=(150, 150, 3), 
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

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range = 180,
                               width_shift_range = 0.1,
                               height_shift_range = 0.1,
                               rescale= 1/255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True,
                               vertical_flip = True,
                               fill_mode = 'nearest'
                               )

test_datagen = ImageDataGenerator(rescale= 1/255)

train_set = train_datagen.flow_from_directory('DATA/train',
                                              target_size=(150, 150),
                                              batch_size=16,
                                              class_mode='binary')


test_set = test_datagen.flow_from_directory('DATA/test',
                                            target_size=(150, 150),
                                            batch_size=16,
                                            class_mode='binary')


train_set.class_indices

results = model.fit_generator(train_set,epochs= 1,
                              steps_per_epoch=150,
                              validation_data=test_set,
                              validation_steps=12)

model.save('model1.h5')

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

Y_pred = model.predict_generator(test_set, 90 // 16+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = ['Glaucoma', 'Healthy']
print(classification_report(test_set.classes, y_pred, target_names=target_names))










