import numpy as np
import keras as K
from keras.utils.data_utils import OrderedEnqueuer
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
import random

from data_generator import DataGenerator

x_train_len = 60000
x_test_len = 10000

x_train_indexes = [i for i in range(x_train_len)]
x_test_indexes = [i for i in range(x_test_len)]

train_params = {'dim': 784,
                'batch_size': 128,
                'n_classes': 10,
                'shuffle': True}

TrainGenerator = DataGenerator(x_train_indexes, dataset='train', **train_params)
TestGenerator = DataGenerator(x_train_indexes, dataset='train', **train_params)

z_dim = 100

adam = Adam(lr=0.0002, beta_1=0.5)

g_input = Input(shape=(z_dim,)) 
g = Dense(256, activation=LeakyReLU(alpha=0.2))(g_input)
g = Dense(512, activation=LeakyReLU(alpha=0.2))(g)
g = Dense(1024, activation=LeakyReLU(alpha=0.2))(g)
g = Dense(784, activation='sigmoid')(g)
generator = Model(g_input, g)
generator.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

d_input = Input(shape=(784,))
d = Dense(1024, activation=LeakyReLU(alpha=0.2))(d_input)
d = Dropout(0.3)(d)
d = Dense(512, activation=LeakyReLU(alpha=0.2))(d)
d = Dropout(0.3)(d)
d = Dense(256, activation=LeakyReLU(alpha=0.2))(d)
d = Dropout(0.3)(d)
d = Dense(1, activation='sigmoid')(d)
descriminator = Model(d_input, d)
descriminator.compile(loss='binary_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

descriminator.trainable = False
gan_input = Input(shape=(z_dim, ))
generated_image = generator(gan_input)
output = descriminator(generated_image)
GAN = Model(gan_input, output)
GAN.compile(loss='binary_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])


def plot_loss(losses):
    """
    @losses.keys():
        0: loss
        1: accuracy
    """
    d_loss = [v[0] for v in losses["D"]]
    g_loss = [v[0] for v in losses["G"]]
    #d_acc = [v[1] for v in losses["D"]]
    #g_acc = [v[1] for v in losses["G"]]
    
    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator loss")
    #plt.plot(d_acc, label="Discriminator accuracy")
    #plt.plot(g_acc, label="Generator accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def plot_generated(n_ex=10, dim=(1, 10), figsize=(12, 2)):
    noise = np.random.normal(0, 1, size=(n_ex, z_dim))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(n_ex, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()





losses = {"D":[], "G":[]}
epochs = 200
steps_per_epoch = np.floor(x_train_len/train_params['batch_size'])
print('steps_per_epoch: ', steps_per_epoch)

enqueuer = OrderedEnqueuer(TrainGenerator,
                           use_multiprocessing=False,
                           shuffle=True)

enqueuer.start(workers=1, max_queue_size=10)
output_generator = enqueuer.get()

for epoch in range(1, epochs+1):
    # if epoch == 1 or epoch % 50 == 0:
    #     print('-'*15,'epoch ',epoch,'-'*15)
    print('-'*15,'epoch ',epoch,'-'*15)
    steps_done = 0
    while steps_done < steps_per_epoch:
        # Create a batch by drawing random index numbers from the training set
        #print('steps_done : ', steps_done)
        generator_output = next(output_generator)
        
        if not hasattr(generator_output, '__len__'):
            raise ValueError('ouput should be tuple (x, y). Found: ' + str(generator_output))
        
        if len(generator_output) == 2:
            X_gen_batch, y_gen_batch = generator_output

            #image_batch = X_gen_batch
            noise = np.random.normal(0, 1, size=(train_params['batch_size'], z_dim))
            generated_images = generator.predict(noise)

            X = np.concatenate((X_gen_batch, generated_images))

            y = np.zeros(2*train_params['batch_size'])
            y[:train_params['batch_size']] = 0.9

            descriminator.trainable = True
            d_loss = descriminator.train_on_batch(X, y)

            noise = np.random.normal(0, 1, size=(train_params['batch_size'], z_dim))
            y2 = np.ones(train_params['batch_size'])
            descriminator.trainable = False
            g_loss = GAN.train_on_batch(noise, y2)

            losses["D"].append(d_loss)
            losses["G"].append(g_loss)

            steps_done += 1

    # Update the plots
    if epoch == 1 or epoch % 20 == 0:
        plot_generated()
        plot_loss(losses)


