import librosa
import numpy as np

# Class that contains the feature computation functions
class Features:
    def __init__(self, n_fft, hop_length, n_mels, num_rows, num_columns):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_rows = num_rows
        self.num_columns = num_columns

    # MFCC
    def MFCC(self, y, sr):
        feat = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.num_rows,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        if feat.shape[1] <= self.num_columns:
            pad_width = self.num_columns - feat.shape[1]
            feat = np.pad(feat, pad_width=((0,0),(0,pad_width)), mode='constant')
            return feat
        else:
            print("mfcc.shape[1]: ", feat.shape[1])
            return None
