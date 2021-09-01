import os
from pydub import AudioSegment

from configs.config import Config
from modules.dataset import features_dataset
from modules.model import CNNModel
from modules.data_generator import DataGenerator


def convert_to_wav(file_path):
    """
    This function is to convert an audio file to .wav file

    Args:
        file_path (str): paths of audio file needed to be convert to .wav file

    Returns:
        new path of .wav file
    """
    ext = file_path.split(".")[-1]
    assert ext in [
        "mp4", "mp3", "acc"], "The current API does not support handling {} files".format(ext)

    sound = AudioSegment.from_file(file_path, ext)
    wav_file_path = ".".join(file_path.split(".")[:-1]) + ".wav"
    sound.export(wav_file_path, format="wav")

    os.remove(file_path)
    return wav_file_path


def predict(df):
    """
    This function is to predict class/probability.

    Args:
        df (dataFrame): include audio path and metadata information.

    Returns:
        assessment (float): class/probability

    """
    df["audio_path"] = df["file_path"]
    df["is_training_set"] = [0] * len(df)
    df["assessment_result"] = [0] * len(df)

    featuresdf = features_dataset(df)
    data = featuresdf.merge(df, left_on='audio_path', right_on='audio_path')
    params = dict(
        batch_size=1,
        n_rows=Config.NUM_ROWS,
        n_columns=Config.NUM_COLUMNS,
        n_channels=Config.NUM_CHANNELS,
    )
    params_test = dict(
        shuffle=False,
        **params
    )
    images = DataGenerator(data, **params_test)

    input_shape = (Config.NUM_ROWS, Config.NUM_COLUMNS, Config.NUM_CHANNELS)

    cnn = CNNModel(input_shape)
    model = cnn.define()
    best_model_path = str(Config.WEIGHT_PATH / "weights.best.hdf5")
    model.load_weights(best_model_path)
    y_pred = model.predict(images)
    return y_pred[0][0]