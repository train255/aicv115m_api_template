import os
import zipfile
import pandas as pd

from configs.config import Config
from modules.dataset import features_dataset
from modules.model import CNNModel
from modules.data_generator import DataGenerator

def create_submission():
    test_df = pd.read_csv(str(Config.ROOT_TEST_DIR / "private_test_sample_submission.csv"))
    test_df['audio_path'] = test_df['uuid'].apply(lambda x: str(
        Config.ROOT_TEST_DIR / f"private_test_audio_files/{x}.wav"))

    test_df["is_training_set"] = [0] * len(test_df)
    test_df["assessment_result"] = [0] * len(test_df)

    test_featuresdf = features_dataset(test_df)
    test_data = test_featuresdf.merge(test_df, left_on='audio_path', right_on='audio_path')
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
    test_images = DataGenerator(test_data, **params_test)

    input_shape = (Config.NUM_ROWS, Config.NUM_COLUMNS, Config.NUM_CHANNELS)

    cnn = CNNModel(input_shape)
    model = cnn.define()
    best_model_path = str(Config.WEIGHT_PATH / "weights.best.hdf5")
    model.load_weights(best_model_path)
    y_pred = model.predict(test_images)

    predictions = [p[0] for p in y_pred]
    test_data["assessment_result"] = predictions
    test_data = test_data.groupby('uuid')['assessment_result'].mean()
    test_data = test_data.reset_index()

    test_df = test_df.drop(columns=["assessment_result"])
    pred_df = pd.merge(test_df, test_data, on="uuid", how="left")
    pred_df["assessment_result"].fillna(0.0, inplace=True)

    pred_df = pred_df[["uuid", "assessment_result"]]
    pred_df.to_csv("results.csv", index=False)

    Config.SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(Config.SUBMISSION_PATH / "results.zip"), 'w') as zf:
        zf.write('results.csv')
    os.remove('results.csv')
