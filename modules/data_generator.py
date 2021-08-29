from tensorflow.keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):
    def __init__(self,
                _X,
                batch_size=32,
                n_channels=1,
                n_columns=470,
                n_rows=120,
                shuffle=True):
        self.batch_size = batch_size
        self.X = _X
        self.n_channels = n_channels
        self.n_columns = n_columns
        self.n_rows = n_rows
        self.shuffle = shuffle
        self.img_indexes = np.arange(len(self.X))
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temps = [self.img_indexes[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temps)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temps):
        X = np.empty((self.batch_size, self.n_rows, self.n_columns))
        y = np.empty((self.batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temps):
            x_features = self.X.iloc[ID]["feature"].tolist()
            label = self.X.iloc[ID]["assessment_result"]
            X[i] = np.array(x_features)
            y[i] = label
        X = X.reshape(X.shape[0], self.n_rows, self.n_columns, self.n_channels)
        return X, y