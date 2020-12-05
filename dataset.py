from torch.utils.data import Dataset
import numpy as np
import keras4torch as k4t

class SpeechCommandsDataset(Dataset):
    def __init__(self, raw_x, raw_y, transform, use_cache=False):
        assert len(raw_x) == len(raw_y)
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.transform = transform
        self.use_cache = use_cache
        self.cache = [None] * len(raw_y)

    def __len__(self):
        return len(self.raw_y)

    def __getitem__(self, index):
        def process():
            x_ = self.raw_x[index]
            y_ = np.array(self.raw_y[index])

            x_ = self.transform(x_).astype('float32')
            return k4t.utils.to_tensor(x_, y_)

        if self.use_cache:
            if self.cache[index] is None:
                self.cache[index] = process()
            return self.cache[index]
        else:
            return process()