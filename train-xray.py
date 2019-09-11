from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array

import keras.backend as K
from keras.backend import tensorflow_backend
import tensorflow as tf
import pandas as pd
import numpy as np

from model.SEResNeXt import SEResNeXt
from utils.img_util import arr_resize
import os
import json
import configparser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def normalize(im):
    im[:, :, 0] = (im[:, :, 0] - 103.94)
    im[:, :, 1] = (im[:, :, 1] - 116.78)
    im[:, :, 2] = (im[:, :, 2] - 123.68)
    return im

## Load parameters
num_classes = 2
batch_size = 16

## Memory setting
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)

## Data preparation

train_data = pd.read_csv('./data/pneumothorax_overmask_train.csv')
# train_data = train_data[0:100]
valid_data = pd.read_csv('./data/pneumothorax_overmask_val.csv')
# valid_data = valid_data[0:100]

IMAGE_SIZE = 512
classes_patologias = ['No Pneumothorax', 'Pneumothorax']

x_train = np.array(
    [normalize(img_to_array(load_img('../hdd/pneumodata/contorno/' + image_name.split('/')[4], target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode='rgb')))
     for image_name in train_data['contours_files'].values])

x_train = arr_resize(x_train, IMAGE_SIZE)

y_train = train_data[classes_patologias]

x_test = np.array(
    [normalize(img_to_array(load_img('../hdd/pneumodata/contorno/' + image_name.split('/')[4], target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode='rgb')))
     for image_name in valid_data['contours_files'].values])

x_test = arr_resize(x_test, IMAGE_SIZE)

y_test = valid_data[classes_patologias]


# y_train = np_utils.to_categorical(y_train, num_classes)
# print(y_train.shape)
# y_test = np_utils.to_categorical(y_test, num_classes)


datagen = ImageDataGenerator(
    rescale = 1/255.
    , shear_range = 0.1
    , zoom_range = 0.1
    , channel_shift_range=0.1
    , rotation_range=15
    , width_shift_range=0.2
    , height_shift_range=0.2
    , horizontal_flip=True)
# datagen.fit(x_train)

valid_datagen = ImageDataGenerator(rescale = 1/255.)
# valid_datagen.fit(x_test)


## Create and compile a model
model = SEResNeXt(IMAGE_SIZE, num_classes).model
sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

def lr_scheduler(epoch):
    if epoch % 30 == 0:
        K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * 0.1)
    return K.eval(model.optimizer.lr)
change_lr = LearningRateScheduler(lr_scheduler)

model.compile(
    optimizer=sgd
    , loss='categorical_crossentropy'
    , metrics=['accuracy'])

## Set callbacks
model_save_name = "./trained_model/SEResNeXt"
filepath = model_save_name + "-{epoch:02d}-{val_acc:.3f}.h5"

csv_logger = CSVLogger('./logs/training.log')
checkpoint = ModelCheckpoint(
    filepath
    , monitor='val_acc'
    , verbose=5
    , save_best_only=True
    , mode='max'
)


## Model training
with open("{0}.json".format(model_save_name), 'w') as f:
    json.dump(json.loads(model.to_json()), f) # model.to_json() is a STRING of json

model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size)
    , steps_per_epoch=len(x_train) // batch_size
    , epochs=100
    , validation_data = valid_datagen.flow(x_test, y_test)
    , validation_steps=len(x_test) // batch_size
    , callbacks=[change_lr, csv_logger, checkpoint])

model.save_weights('{0}.h5'.format(model_save_name))
model.save('{0}_archweights.h5'.format(model_save_name))
