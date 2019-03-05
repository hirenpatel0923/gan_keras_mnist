import numpy as np
import keras as K
from keras.utils.data_utils import OrderedEnqueuer
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Lambda, concatenate
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


x_dim = 784
z_dim = 100
y_dim = 10

adam = Adam(lr=0.0002, beta_1=0.5)



#generator model
g_input_zdim = Input(shape=(z_dim,))  
g_input_label = Input(shape=(y_dim,))
g_input = concatenate([g_input_zdim, g_input_label])
g = Dense(256, activation=LeakyReLU(alpha=0.2))(g_input)
g = Dense(512, activation=LeakyReLU(alpha=0.2))(g)
g = Dense(1024, activation=LeakyReLU(alpha=0.2))(g)
g = Dense(784, activation='sigmoid')(g)
generator = Model([g_input_zdim, g_input_label], g)
generator.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

#discriminator model real/fake
d_input = Input(shape=(x_dim,))
#d_input = concatenate([d_input_image, input_label])
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

#discriminator model for 0-9
d2_input_img = Input(shape=(x_dim,))
d2_input_label = Input(shape=(y_dim,))
d2_input = concatenate([d2_input_img, d2_input_label])
d2 = Dense(1024, activation=LeakyReLU(alpha=0.2))(d2_input)
d2 = Dropout(0.3)(d2)
d2 = Dense(512, activation=LeakyReLU(alpha=0.2))(d2)
d2 = Dropout(0.3)(d2)
d2 = Dense(256, activation=LeakyReLU(alpha=0.2))(d2)
d2 = Dropout(0.3)(d2)
d2 = Dense(y_dim, activation='sigmoid')(d2)
descriminator2 = Model([d2_input_img, d2_input_label], d2)
descriminator2.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
#print(descriminator2.summary())

#gan model
descriminator.trainable = False
descriminator2.trainable = False
gan_input_zdim = Input(shape=(z_dim,))
gan_input_label = Input(shape=(y_dim,))
generated_image = generator([gan_input_zdim, gan_input_label])
d1_output = descriminator(generated_image)
d2_output = descriminator2([generated_image,gan_input_label])
GAN = Model([gan_input_zdim, gan_input_label], [d1_output, d2_output])
GAN.compile(loss='binary_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])

#print(GAN.summary())


def get_conditional_batch(x, y, x_dim, y_dim):
    #y = K.utils.to_categorical(y, num_classes=10)
    X_new = np.zeros((x.shape[0], (x_dim + y_dim)))
    for ID, X in enumerate(X_new):
        X_new[ID] = np.concatenate([x[ID], y[ID]])

    return X_new
    


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


def show_specific_num_image(num, z_dim):
    y_tmp = K.utils.to_categorical(num, num_classes=10)
    y_tmp = np.reshape(y_tmp, (1, y_tmp.shape[-1]))
    n = np.random.normal(0, 1, size=(1, z_dim))

    # z_dim_concate = np.concatenate([n, y_tmp])
    # z_dim_concate = np.reshape(z_dim_concate, (1, z_dim_concate.shape[-1]))
    
    gen_image = generator.predict([n, y_tmp])

    print('showing image of : ',num)
    plt.imshow(gen_image[0].reshape([28, 28]), cmap='Greys')
    plt.show()



losses = {"D":[], "G":[]}
epochs = 200
steps_per_epoch = np.floor(x_train_len/train_params['batch_size'])


#training process
orderedEnqueuer = OrderedEnqueuer(TrainGenerator,
                                  use_multiprocessing=False,
                                  shuffle=True)

orderedEnqueuer.start(workers=1, max_queue_size=10)
output_generator = orderedEnqueuer.get()


#training process
for epoch in range(1, epochs+1):
    print('-'*15, ' epoch ',epoch,' ','-'*15)
    steps_done = 0
    while steps_done < steps_per_epoch:
        generator_opt = next(output_generator)

        if len(generator_opt) == 2:
            x, y = generator_opt

            noise = np.random.normal(0, 1, size=(train_params['batch_size'], z_dim))
            generated_images = generator.predict([noise, y])

            #des1 training
            X = np.concatenate([x, generated_images])
            y_des = np.zeros(2*train_params['batch_size'])
            y_des[:train_params['batch_size']] = 1
            descriminator.trainable = True
            d1_loss = descriminator.train_on_batch(X, y_des)

            #des2 training
            descriminator2.trainable = True
            d2_loss = descriminator2.train_on_batch([x, y], y)

            d_loss = d1_loss + d2_loss

            noise = np.random.normal(0, 1, size=(train_params['batch_size'], z_dim))
            y_des1 = np.ones(train_params['batch_size'])
            #y_des2 = np.ones((train_params['batch_size'], y_dim))
            descriminator.trainable = False
            descriminator2.trainable = False
            g_loss = GAN.train_on_batch([noise, y], [y_des1, y])

            losses["D"].append(d_loss)
            losses["G"].append(g_loss)

            steps_done += 1

    generator.save_weights('trained_model/cgan_generator.h5')
    descriminator.save_weights('trained_model/cgan_desciminator.h5')
    descriminator2.save_weights('trained_model/cgan_desciminator2.h5')
    GAN.save_weights('trained_model/cgan_GAN.h5')



    # Update the plots
    if epoch == 1 or epoch % 20 == 0:
        #plot_generated()
        plot_loss(losses)
        show_specific_num_image(random.randint(0,9), z_dim)

