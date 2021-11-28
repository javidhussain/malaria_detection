import numpy as np 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

path = '/home/superman/Project/Convolution neural networks/cell_images/'
data_gen=ImageDataGenerator(rescale=1/255.0,validation_split=0.2)
train_gen=data_gen.flow_from_directory(directory=path,target_size=(50,50),class_mode='binary',batch_size=16,subset='training')
val_gen=data_gen.flow_from_directory(directory=path,target_size=(50,50),class_mode='binary',batch_size=16,subset='validation')


model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(50,50,3)))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#early_stop = EarlyStopping(monitor='val_loss',patience=2)

#history=model.fit_generator(generator=train_gen,steps_per_epoch=len(train_gen),epochs=2,validation_steps=len(val_gen),validation_data=val_gen,callbacks=early_stop)
model.fit_generator(generator=train_gen,steps_per_epoch=len(train_gen),epochs=2,validation_steps=len(val_gen),validation_data=val_gen)
model.save('Malaria')

