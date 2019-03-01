import numpy as np
from data_generator import DataGenerator
from keras.utils.data_utils import OrderedEnqueuer
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
import random

x_train_indexes = [i for i in range(60000)]
x_test_indexes = [i for i in range(10000)]

train_params = {'dim': 784,
                   'batch_size': 128,
                   'n_classes': 10,
                   'shuffle': True}

steps_per_epoch = np.floor(60000/train_params['batch_size'])

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


enqueuer = OrderedEnqueuer(TrainGenerator,
                           use_multiprocessing=False,
                           shuffle=False
                           )

enqueuer.start(workers=1, max_queue_size=10)
output_generator = enqueuer.get()

for i in range(1):
    steps_done = 0
    while steps_done < steps_per_epoch:
        print('batch: ', steps_done)
        generator_output = next(output_generator)
        if len(generator_output) == 2:
            x, y = generator_output
            model.train_on_batch(x, y)
        else:
            print('generator output length : ', len(generator_output))
        steps_done += 1

loss, acc = model.evaluate_generator(TestGenerator, steps=3, verbose=0)

print(acc)
print(loss)