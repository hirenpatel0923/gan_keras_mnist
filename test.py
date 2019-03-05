import keras as K
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Lambda, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

x_dim = 784
z_dim = 100
y_dim = 10

adam = Adam(lr=0.0002, beta_1=0.5)

input_label = Input(shape=(y_dim,))

#generator model
g_input_zdim = Input(shape=(z_dim,))  
g_input = concatenate([g_input_zdim, input_label])
g = Dense(256, activation=LeakyReLU(alpha=0.2))(g_input)
g = Dense(512, activation=LeakyReLU(alpha=0.2))(g)
g = Dense(1024, activation=LeakyReLU(alpha=0.2))(g)
g = Dense(784, activation='sigmoid')(g)
generator = Model([g_input_zdim, input_label], g)
generator.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])


generator.load_weights('trained_model/cgan_generator.h5')

#num = 7
for num in range(0, 10):
    y_tmp = K.utils.to_categorical(num, num_classes=10)
    y_tmp = np.reshape(y_tmp, (1, y_tmp.shape[-1]))
    n = np.random.normal(0, 1, size=(1, z_dim))

    gen_image = generator.predict([n, y_tmp])

    print('showing image of : ',num)
    plt.imshow(gen_image[0].reshape([28, 28]), cmap='Greys')
    plt.show()