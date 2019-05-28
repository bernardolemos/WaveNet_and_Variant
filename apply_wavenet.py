import numpy as np
from data_helpers import *
from sklearn.model_selection import train_test_split
from wavenet_and_variant import *


def train_wavenet(data_file, max_len=2**22):
    """
    :data_file - *.npy file with a collection of arrays (2D array)
    :max_len - the sequence length to feed the network
    """
    #read data
    data = np.load(data_file, encoding='bytes', allow_pickle=True)
    #merge everything
    X = np.empty((0,))
    for d in data:
        X = np.concatenate([X, d])
    #convert to 3D
    X_chunk, y = chunk_data(max_len, X)
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X_chunk, y, test_size=0.3, shuffle=False)
    #build wavenet
    wavenet = wavenet(X_train.shape[1], summary=False)
    wavenet.fit(X_train, y_train, batch_size=16, shuffle=False)

    return wavenet


def main():
    data_file = "./data.npy"
    max_len = 2**22
    #train model
    # model_wn = train_wavenet(data_file)
    #apply model
    X_fake = gen_fake_data(256)
    model = wavenet(256, summary=False)
    X_gen = gen_data_from_model(model, 256, X_fake, hard=True)


if __name__ == "__main__":
    main()