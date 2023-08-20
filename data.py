import os
from torch.utils.data import Dataset
import librosa
import librosa.display
import numpy as np


class VoiceDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.segment_files = [
            file for file in os.listdir(data_folder) if file.endswith(".mp3")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.segment_files)

    def __getitem__(self, idx):
        segment_file = self.segment_files[idx]
        segment_path = os.path.join(self.data_folder, segment_file)

        y, sr = librosa.load(segment_path, sr=None)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8000)
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

        if self.transform:
            mfccs = self.transform(mfccs)

        return mfccs
