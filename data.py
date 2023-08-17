import os
from torch.utils.data import Dataset
from pydub import AudioSegment


class VoiceDataset(Dataset):
    def __init__(self, data_folder, fixed_duration_ms, transform=None):
        self.data_folder = data_folder
        self.segment_files = [
            file for file in os.listdir(data_folder) if file.endswith(".mp3")
        ]
        self.fixed_duration_ms = fixed_duration_ms
        self.transform = transform

    def __len__(self):
        return len(self.segment_files)

    def __getitem__(self, idx):
        segment_file = self.segment_files[idx]
        segment_path = os.path.join(self.data_folder, segment_file)

        audio = AudioSegment.from_mp3(segment_path)
        audio = audio.set_channels(1)

        if len(audio) > self.fixed_duration_ms:
            audio = audio[: self.fixed_duration_ms]
        elif len(audio) < self.fixed_duration_ms:
            padding = self.fixed_duration_ms - len(audio)
            silence = AudioSegment.silent(duration=padding)
            audio = silence + audio

        if self.transform:
            audio = self.transform(audio)

        return audio
