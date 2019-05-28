import numpy as np
from data_helpers import *
from scipy.io.wavfile import read, write
from sklearn.model_selection import train_test_split
from wavenet_and_variant import *

##Read data
DATA_FILE = "./data.npy"
LEFT_FRAME = 2 ** 22
data = np.load(DATA_FILE, encoding='bytes', allow_pickle=True)
#has data.shape[0] observations
data = np.array(data).flatten()
print(data)