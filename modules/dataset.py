import numpy as np
import pandas as pd
import librosa
import aubio

# from scipy.io import wavfile

from sklearn.model_selection import train_test_split

from modules.feature import Features
from modules.data_generator import DataGenerator


# def read_wav_file(filepath):
#     sample_rate, waveform = wavfile.read(filepath)
#     if waveform.ndim > 1:
#         waveform = waveform[:, 0]
#     return sample_rate, waveform


def avg_fq(audio_path):
    win_s = 2048
    hop_s = win_s // 4

    s = aubio.source(audio_path, hop_s)
    tolerance = 0.8
    pitch_o = aubio.pitch("yin", win_s, hop_s, s.samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)
    pitches = []

    total_frames = 0
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        pitches += [pitch]
        total_frames += read
        if read < hop_s: break
    a = np.array(pitches)
    return a.mean()


def audio_augmentation(samples):
    y_aug = samples.copy()
    dyn_change = np.random.uniform(low=1.5,high=3)
    y_aug = y_aug * dyn_change

    y_noise = samples.copy()
    noise_amp = np.random.uniform(0.001, 0.005) * np.amax(y_noise)
    y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])

    y_shift = samples.copy()
    timeshift_fac = 0.1 *2*(np.random.uniform()-0.5)
    start = int(y_shift.shape[0] * timeshift_fac)
    if (start > 0):
        y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]
    else:
        y_shift = np.pad(y_shift,(0,-start),mode='constant')[0:y_shift.shape[0]]

    y_hpss = librosa.effects.hpss(samples.astype('float64'))

    return [y_aug, y_noise, y_shift, y_hpss[1]]


def splitAudio(y, audio_path):
    nonMuteSections = librosa.effects.split(y, 21)
    if (len(nonMuteSections) == 1) and (y.shape[0] == nonMuteSections[0][1]):
        frq_mean = avg_fq(audio_path)
        if frq_mean < 1:
            nonMuteSections = []

    wave_no_silents = []
    if len(nonMuteSections) > 0:    
        for i in nonMuteSections:
            non_mute_wave = y[i[0]:i[1]]
            wave_no_silents.append(non_mute_wave)    
    return wave_no_silents


def extract_features(file_name, is_training_set, covid):
    try:
        audio, sample_rate = librosa.load(file_name)
        # sample_rate, audio = read_wav_file(file_name)
        waves = []
        audio_splits = splitAudio(audio, file_name)
        num_rows = 120
        num_columns = 430
        n_fft = 4096
        hop_length = n_fft // 4
        # hop_length = 512
        n_mels = 512

        extractor = Features(n_fft, hop_length, n_mels, num_rows, num_columns)
        if len(audio_splits) > 0:
            for au_clean in audio_splits:
                audio_aug = [au_clean]

                if is_training_set == 1 and covid == 1:
                    audio_aug = audio_aug + audio_augmentation(au_clean)

                for au_aug in audio_aug:
                    audio_features = extractor.MFCC(au_aug, sample_rate)
                    if audio_features is not None:
                        waves.append(audio_features)
    except Exception as e:
        print("Error encountered while parsing file: ", e)
        return []

    return waves


def features_dataset(df):
    features = []
    for index, row in df.iterrows():
        file_name = row["audio_path"]
        is_training_set = row["is_training_set"]
        covid = row["assessment_result"]
        features_lst = extract_features(file_name, is_training_set, covid)
        if len(features_lst) > 0:
            for data in features_lst:
                features.append([data, file_name])
        else:
            print("Data is empty: ", file_name)

    featuresdf = pd.DataFrame(features, columns=['feature', 'audio_path'])
    print('Finished feature extraction from ', len(featuresdf), ' files')
    num_rows = featuresdf[featuresdf.index == 0].feature[0].shape[0]
    num_columns = featuresdf[featuresdf.index == 0].feature[0].shape[1]
    num_channels = 1
    input_shape = (num_rows, num_columns, num_channels)

    return featuresdf, input_shape


def get_train_data(train_meta_df, train_extra_df):
    # Not cough sound
    train_meta_df.drop(train_meta_df[train_meta_df['uuid'] == "23ccaa28-8cb8-43e4-9e59-112fa4dc6559"].index, inplace = True)
    train_nonote_df = train_meta_df[train_meta_df.audio_noise_note.isnull()].reset_index(drop=True)
    train_nonote_df = train_nonote_df[["uuid","assessment_result", "audio_path", "cough_intervals"]]

    train_covid_extra_df = train_extra_df[train_extra_df['assessment_result'] == "1"].reset_index(drop=True)
    # File not found
    train_covid_extra_df = train_covid_extra_df.drop([10]).reset_index(drop=True)
    train_covid_extra_df["assessment_result"] = train_covid_extra_df["assessment_result"].apply(lambda x: int(x))
    # Train Dataset
    train_data = pd.concat([train_nonote_df, train_covid_extra_df]).reset_index(drop=True)
    # Train test split
    idx_train, _, _, _ = train_test_split(train_data[["audio_path"]], train_data[["assessment_result"]], test_size=0.1, random_state = 42)
    path_lst = idx_train["audio_path"].tolist()
    train_data["is_training_set"] = train_data["audio_path"].apply(lambda x: 1 if x in path_lst else 0)
    # Get features
    train_featuresdf, input_shape = features_dataset(train_data)
    train_data = train_data.merge(train_featuresdf, left_on='audio_path', right_on='audio_path')

    # Data generator
    params = dict(
        batch_size=16,
        n_rows=input_shape[0],
        n_columns=input_shape[1],
        n_channels=input_shape[2],
    )
    params_train = dict(
        shuffle=True,
        **params
    )
    params_valid = dict(
        shuffle=False,
        **params
    )
    X_train = train_data[train_data["is_training_set"] == 1].reset_index(drop=True)
    X_valid = train_data[train_data["is_training_set"] == 0].reset_index(drop=True)
    train_generator = DataGenerator(X_train, **params_train)
    valid_generator = DataGenerator(X_valid, **params_valid)

    return train_generator, valid_generator, input_shape