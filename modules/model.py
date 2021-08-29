from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D

class CNNModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def define(self):
        # Define model
        input_shape = self.input_shape
        model = Sequential()
        model.add(Conv2D(16, (7,7), input_shape=input_shape, activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3), activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3,3), activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(512, (1,1), activation='relu', padding="same"))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1, activation='sigmoid'))
        return model