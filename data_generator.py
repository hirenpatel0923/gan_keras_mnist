import numpy as np
import keras
from keras.datasets import mnist

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dataset='train', dim=100, batch_size=32, n_classes=1, shuffle=False):
        'Initialization'
        self.dim = dim
        self.dataset = 'train'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        #get dataset
        self.x_train, self.y_train, self.x_test, self.y_test = self.__prepare_dataset()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            if self.dataset == 'train':
                # Store sample
                X[i,] = self.x_train[ID]

                # Store class
                y[i] = self.y_train[ID]
            elif self.dataset == 'test':
                 # Store sample
                X[i,] = self.x_test[ID]

                # Store class
                y[i] = self.y_test[ID]
            else:
                ValueError('invalid dataset...!')

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __prepare_dataset(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32')/255

        return x_train, y_train, x_test, y_test
# class Generator():
#     def generator(self):
#         lst = range(self.get_lst())
#         for i in lst:
#             yield i, i*i
#     def get_lst(self):
#         return 5

# myGenerator = Generator()

# for i, sqr in myGenerator.generator():
#     print(i)
#     print(sqr)
#     print('-'*15)