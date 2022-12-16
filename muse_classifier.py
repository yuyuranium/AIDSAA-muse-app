import random
import librosa
import numpy as np
from tensorflow.keras.models import Model, load_model

class MuseClassifier():
    def __init__(self, model_path):
        # self.model = keras.load(model_path)
        self.sr = 22050
        print('Loading model from %s' % model_path)
        self.model = load_model(model_path)
        print('Done loading mdoel')

    def gen_aug_feature(self, raw_data):
        # MFCC
        mfcc = librosa.feature.mfcc(
                y=raw_data, sr=self.sr, hop_length=self.sr, n_mfcc=1)
        mfcc = mfcc.reshape((1, mfcc.shape[1] * mfcc.shape[0]))

        # Roll-off
        rolloff = librosa.feature.spectral_rolloff(
                y=raw_data, sr=self.sr, hop_length=self.sr, roll_percent=0.85)

        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
                y=raw_data, sr=self.sr, hop_length=self.sr, n_bands=1)
        spectral_contrast = spectral_contrast.reshape(
                (1, spectral_contrast.shape[1] * spectral_contrast.shape[0]))

        # RMS
        rms = librosa.feature.rms(y=raw_data, hop_length=self.sr)

        # Combine
        combine = np.hstack((rms, mfcc))
        combine = np.hstack((combine, rolloff))
        combine = np.hstack((combine, spectral_contrast))

        return combine

    def predict(self, raw_data):
        print('Predicting...')
        x_audio, x_feat = np.expand_dims(raw_data, axis=0), self.gen_aug_feature(raw_data)

        pred = self.model.predict([x_audio, x_feat])[0]
        print(pred)

        print('Done predicting')
        return np.argmax(pred)
