import pandas as pd

from configs.config import Config
from modules.dataset import get_train_data

from modules.model import CNNModel

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

def prepare_training_data(Config):
    train_meta_df = pd.read_csv(str(Config.ROOT_TRAIN_DIR / "public_train_metadata.csv"))
    train_meta_df['audio_path'] = train_meta_df['uuid'].apply(lambda x: str(
        Config.ROOT_TRAIN_DIR / f"public_train_audio_files/{x}.wav"))

    train_extra_df = pd.read_csv(str(Config.ROOT_EXTRA_TRAIN_DIR / "extra_public_train_1235samples.csv"))
    train_extra_df['audio_path'] = train_extra_df['uuid'].apply(lambda x: str(
        Config.ROOT_EXTRA_TRAIN_DIR / f"new_1235_audio_files/{x}.wav"))

    return get_train_data(train_meta_df, train_extra_df)


def train():
    best_model_path = str(Config.WEIGHT_PATH / "weights.best.basic_cnn_mfcc.hdf5")
    train_generator, valid_generator, input_shape = prepare_training_data(Config)

    cnn = CNNModel(input_shape)
    model = cnn.define()
    
    opt = Adam(learning_rate=0.00001)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    checkpointer = ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy", verbose=1, save_best_only=True)
    es_callback = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1)

    model.fit(
        train_generator,
        epochs=100,
        validation_data=valid_generator,
        callbacks=[checkpointer,es_callback],
        verbose=1
    )


if __name__ == "__main__":
    train()
