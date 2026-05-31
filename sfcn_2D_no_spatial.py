import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, Conv2D,MaxPool2D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling2D, AveragePooling3D, LeakyReLU, Add
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, SGD, Nadam
from tensorflow.keras.regularizers import l1_l2, l1, l2
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import argparse
import sys


#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-x_train', type=str, help='training set')
parser.add_argument('-y_train', type=str, help='training set')
parser.add_argument('-x_val', type=str, help='training set')
parser.add_argument('-y_val', type=str, help='training set')
parser.add_argument('-out_size_x', type=int, help='training set size')
parser.add_argument('-out_size_y', type=int, help='training set size')
parser.add_argument('-in_size', type=int, help='sample set size')
parser.add_argument('-out_name', type=str, help='name to save')
args = parser.parse_args()

def sfcn(inputLayer):
    #block 1
    x=Conv2D(filters=32, kernel_size=(1,2),strides=(1, 1), padding='same',name="conv1")(inputLayer[0])
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool2D(pool_size=(2, 1),strides=(2, 1),padding='same',name="maxpool1")(x)
    x=ReLU()(x)

    #block 2
    x=Conv2D(filters=64, kernel_size=(1, 2),strides=(1, 1),padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool2D(pool_size=(2, 1),strides=(2, 1),padding='same',name="maxpool2")(x)
    x=ReLU()(x)

    #block 3
    x=Conv2D(filters=128, kernel_size=(1, 2),strides=(1, 1),padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool2D(pool_size=(2, 1),strides=(2, 1),padding='same',name="maxpool3")(x)
    x=ReLU()(x)

    #block 4
    x=Conv2D(filters=256, kernel_size=(1, 2),strides=(1, 1),padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool2D(pool_size=(2, 1),strides=(2, 1),padding='same',name="maxpool4")(x)
    x=ReLU()(x)

    #block 5
    x=Conv2D(filters=256, kernel_size=(1, 2),strides=(1, 1),padding='same',name="conv5")(x)
    x=BatchNormalization(name="norm5")(x)
    x=MaxPool2D(pool_size=(2, 1),strides=(2, 1),padding='same',name="maxpool5")(x)
    x=ReLU()(x)


    #block 6
    x=Conv2D(filters=64, kernel_size=(1, 1),padding='same',name="conv7")(x)
    x=BatchNormalization(name="norm7")(x)
    x=ReLU()(x)


    #block 7, different from paper
    x=AveragePooling2D(padding='same')(x)
    x=Dropout(.5)(x)
    x = Flatten(name="flat1")(x)
    x=Dense(units=1, activation='sigmoid',name="dense1")(x)
    return x

def compile_model(in_size):
    opt = Adam(learning_rate=0.0001)
    metr = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    inputA = Input(shape=(in_size, 32, 1), name="InputA") #1000 or 100 or 10
    z = sfcn([inputA])  # Ensure this outputs a single value for classification
    print(z)
    
    model = Model(inputs=[inputA], outputs=[z])
    model.summary()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=metr)
    
    return model

def scheduler(epoch, lr):
    if epoch%2==0: #was5
      return lr * tf.math.exp(-0.1)
    else:
      return lr
    
x_train = np.load(args.x_train)
y_train = np.load(args.y_train)
x_val = np.load(args.x_val)
y_val = np.load(args.y_val)
out_size_x = args.out_size_x
out_size_y = args.out_size_y
in_size = args.in_size
checkpoint_path = args.out_name+"_best.h5"

# Replace true labels with random binary labels
# y_train = np.random.randint(0, 2, size=y_train.shape)
# y_val = np.random.randint(0, 2, size=y_val.shape)

print(x_train.shape)
x_train = x_train.reshape((out_size_x, in_size, 32, 1)) #1000 or 100 or 10
x_val = x_val.reshape((out_size_y, in_size, 32, 1))  # Adjust for validation set


model=compile_model(in_size)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to watch
    patience=10,          # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restores best weights after stopping
)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=2,
                                                         save_best_only=True, include_optimizer=True,
                                                         save_weights_only=False, mode='auto',
                                                         save_freq='epoch')


history = model.fit(x=x_train,y=y_train, callbacks=[lr_callback,checkpoint_callback, earlystop], validation_data=(x_val, y_val), batch_size=64, epochs=100, shuffle=True, verbose=2)

history_dict = history.history