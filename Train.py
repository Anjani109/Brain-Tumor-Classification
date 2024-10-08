import cv2
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import utils
utils.to_categorical
utils.normalize
from keras import models 
models.Sequential
from keras import layers
layers.Conv2D
layers.MaxPooling2D
layers.Activation
layers.Dropout
layers.Flatten
layers.Dense
from keras import metrics
metrics.Accuracy


image_directory = 'Datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

dataset = []
label = []

INPUT_SIZE = 64

# 1) Separate and Resize Images (YES and NO)
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

# 2) Converting Data into NUMPY Array
dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)
# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)
x_train = utils.normalize(x_train, axis=1)
x_test = utils.normalize(x_test, axis=1)

# y_train = utils.to_categorical(y_train, num_classes=2)
# y_test=utils.to_categorical(y_test,num_classes=2)

# 3) Import Models
# 4) Model Building
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=16,
          verbose=1, epochs=10,
          validation_data=(x_test, y_test),
          shuffle=False)

model.save('BrainTumor10Epoch.h5')
