from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

class CNNModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def define(self):
        img_input = Input(shape=self.input_shape)
        img_conc = Concatenate()([img_input, img_input, img_input])
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=img_conc
        )
        # base_model.trainable = False
        avgpool = GlobalAveragePooling2D()(base_model.output)
        # outputs = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='sigmoid')(avgpool)
        outputs = Dense(1, activation='sigmoid')(avgpool)

        model = Model(inputs=base_model.input, outputs=outputs)
        return model