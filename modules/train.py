import pandas as pd

from configs.config import Config
from modules.dataset import get_train_data

from modules.model import CNNModel

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
    best_model_path = str(Config.WEIGHT_PATH / "weights.best.hdf5")
    train_generator, valid_generator = prepare_training_data(Config)
    input_shape = (Config.NUM_ROWS, Config.NUM_COLUMNS, Config.NUM_CHANNELS)
    cnn = CNNModel(input_shape)
    model = cnn.define()
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=["accuracy"])

    checkpointer = ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy", verbose=1, save_best_only=True)
    es_callback = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=1, verbose=1, min_delta=0.0001, cooldown=1, min_lr=0.00001)

    model.fit(
        train_generator,
        epochs=50,
        validation_data=valid_generator,
        callbacks=[checkpointer,es_callback,reduce_lr],
        verbose=1
    )


if __name__ == "__main__":
    train()
