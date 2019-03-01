import numpy as np
import keras as K
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
import random

from data_generator import DataGenerator

x_train_indexes = [i for i in range(60000)]
x_test_indexes = [i for i in range(10000)]

train_params = {'dim': 784,
                   'batch_size': 64,
                   'n_classes': 10,
                   'shuffle': True}

TrainGenerator = DataGenerator(x_train_indexes, dataset='train', **train_params)
TestGenerator = DataGenerator(x_train_indexes, dataset='train', **train_params)

i = Input(shape=(784,))
o = Dense(1024, activation='relu')(i)
o = Dense(512, activation='relu')(o)
o = Dense(256, activation='relu')(o)
o = Dense(64, activation='relu')(o)
o = Dense(10, activation='softmax')(o)
model = Model(i, o)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(generator=TrainGenerator,
                    epochs=1)

#predict = model.predict_generator(generator=TestGenerator)
loss, acc = model.evaluate_generator(TestGenerator, steps=3, verbose=0)
print(acc)
print(loss)