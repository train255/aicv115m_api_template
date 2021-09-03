import os
from pydub import AudioSegment

from configs.config import Config
from modules.dataset import features_dataset
from modules.model import CNNModel
import numpy as np

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

    featuresdf, input_shape = features_dataset(df)
    data = featuresdf.merge(df, left_on='audio_path', right_on='audio_path')
    images = np.array(data.feature.tolist())
    images = images.reshape(images.shape[0], input_shape[0], input_shape[1], input_shape[2])

    cnn = CNNModel(input_shape)
    model = cnn.define()
    best_model_path = str(Config.WEIGHT_PATH / "weights.best.basic_cnn_mfcc.hdf5")
    model.load_weights(best_model_path)
    y_pred = model.predict(images)
    
    predictions = [p[0] for p in y_pred]
    return np.mean(predictions)