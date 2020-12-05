from abc import abstractclassmethod
import numpy as np
import random
from torchvision.transforms import *
import librosa as lb

SR = 16000

train_mel32_stats = (-34.256863, 17.706715)
train_mel64_stats = (-41.177162, 20.649311)

def normalize(x, eps=1e-6):
    x = (x - np.mean(x)) / (np.std(x) + eps)
    if x.max() - x.min() < eps:
        x = np.zeros_like(x)
    return x.astype('float32')

class GlobalNormalization(object):
    def __init__(self, config):
        assert config == 'mel32' or config == 'mel64'
        if config == 'mel32':
            self.mean, self.std = train_mel32_stats
        if config == 'mel64':
            self.mean, self.std = train_mel64_stats
    
    def __call__(self, x):
        return (x - self.mean) / self.std

def crop_or_pad(y, length=16000):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    elif len(y) > length:
        y = y[: length]
    return y

class ToLogMelspectrogram(object):
    def __init__(self, config):
        if config == '1x32x32':
            self.hop_length, self.n_fft, self.n_mels = 512, 2048, 32
            self.pad = lambda x: x
        elif config == '1x64x64':
            self.hop_length, self.n_fft, self.n_mels = 256, 1024, 64
            # [64, 63] -> [64, 64]
            self.pad = lambda x: np.c_[x, np.zeros([64])]

    def __call__(self, x):
        assert len(x) == 16000
        melspec = lb.feature.melspectrogram(x, sr=SR, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
        melspec = lb.power_to_db(melspec)
        return self.pad(melspec)[np.newaxis, :, :]



class StochasticTransform(object):
    def __call__(self, x):
        return self.transform(x) if random.random() < 0.5 else x

    @abstractclassmethod
    def transform(self, x):
        pass

class ChangeAmplitude(StochasticTransform):
    def __init__(self, min=0.7, max=1.2):
        self.amplitude_range = (min, max)

    def transform(self, x):
        x = x * random.uniform(*self.amplitude_range)
        return x
    
class ChangeSpeedAndPitch(StochasticTransform):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def transform(self, x):
        scale = random.uniform(-self.max_scale, self.max_scale)
        speed_fac = 1.0  / (1 + scale)
        x = np.interp(np.arange(0, len(x), speed_fac), np.arange(0,len(x)), x).astype(np.float32)
        return x

class TimeShift(StochasticTransform):
    def __init__(self, frac_0=8, frac_1=3):
        self.frac_0 = frac_0
        self.frac_1 = frac_1

    def transform(self, x):
        a = np.arange(len(x))
        a = np.roll(a, np.random.randint(len(x)//self.frac_0, len(x)//self.frac_1))
        return x[a]