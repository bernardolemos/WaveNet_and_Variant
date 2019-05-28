import numpy as np

"""
This a collection of functions to generate fake data to feed
WavNet model and its variant.
It also includes a function to generate data from WaveNet
"""

########### Data Helpers #################################################
##########################################################################
##### Synthetic data generators ##########################################
def gen_random_labels(n_chunks):
    """
    Generate random distribuiton [0, 1] for:
        p(speech), p(non-speech)
    """
    y = np.zeros(shape=(n_chunks, 2))
    for i in range(n_chunks):
        p_speech = np.random.uniform(0,1)
        y[i, 0] = p_speech
        y[i, 1] = 1 - p_speech

    return y

def gen_fake_data(n, min_v=0, max_v=256):
    """
    Generate random integer values following a uniform distribution  
    """
    # return np.random.uniform(min_v, max_v, n)
    return np.random.randint(min_v, max_v, n)
#####################################

def chunk_data(left_frames, data):
    """
    Reshapes data (input) for network. 
    3D tensor: (#observation, sequence len., 1)
    Params.
    :left_frames - sequence length
    :data - 1d np.array
    """

    rs = data.shape[0] % left_frames
    n_chunks = data.shape[0] // left_frames
    #exclude values
    if rs > 0:
        data = data[:-rs]
    #build data
    X = data.reshape((n_chunks, left_frames, 1)).astype(int)
    #build labels
    y = np.zeros(shape=(n_chunks, left_frames))
    for i in range(n_chunks):
        #get last value from chunk i
        y[i, X[i, -1, 0]] = 1 

    return X, y

########### Data generator for WaveNet ###################################
def gen_data_from_model(model, out_size, X, hard=True):
    """
    Generate data (audio) from input, using WaveNet model

    Parameters
    :model - a WaveNet model
    :out_size - the size og the output
    :X - initial data, 1D np.array. Model generated data from X's values
    :hard - boolean. The way the generated value is chosen.
                     True if max. value from softmax is to be used, 
                     otherwise, use predictions' ditribution
    """
    #init. output
    gen_out = np.zeros(out_size, dtype=int)
    #reshape input to 3D array (model's input shape)
    X = X.reshape(1, X.shape[0], 1)
    #generate
    print("Generating data from model...")
    for i in range(out_size):
        #get model's predictions
        predictions = model.predict(X)
        #reshape predictions to 256 (predictions shape == input shape)
        predictions_256 = predictions.reshape(256)
        #get generated value
        if hard:
            gen_val = np.argmax(predictions_256)
        else:
            gen_val = np.random.choice(range(256), p=predictions_256)
        
        #amplify!
        ampl_val_8 = ((((gen_val) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(ampl_val_8) - 1)) * 2**15
        ampl_val_16 = int(ampl_val_16) #cast
        
        #update generated data, add value
        gen_out[i] = ampl_val_16
        #update model's input
        #remove first
        X[:-1] = X[1:]
        #add generated value to end of array
        X[-1] = ampl_val_16    
    
    return gen_out