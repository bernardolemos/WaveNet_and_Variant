"""
1. WaveNet: : A Generative Model for Raw Audio - https://arxiv.org/pdf/1609.03499.pdf
2. Temporal Modeling Using Dilated Convolution and Fating for Voice-Activity-Detection 
"""

import numpy as np
from wavenet_block import wavenet_block
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Multiply, Add, Flatten, Dense, Input, Activation

def wavenet(input_dim, k_layers=20, max_dil_rate=512, n_atrous_filters=64, atrous_filter_size = 2, 
                loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], summary=True):
    """
    Buil WaveNet network

    Netowrk's input observations are arrays with #`input_dim` of 1D arrays

    Params.
    :input_dim - # of steps
    :k_layers - # of layers in stack
    ...

    Return
    :keras model
    """
    #Input tensor
    input_tensor = Input(shape=(input_dim, 1)) 
    ##Causal convolution before k Layers
    causal_input = Conv1D(1, atrous_filter_size, padding='causal', data_format='channels_last', dilation_rate=1, activation='linear')(input_tensor)
    ##First block 
    atrous_rate = 2
    res, skip = wavenet_block(n_atrous_filters, atrous_filter_size, atrous_rate)(causal_input) # atrous deveria ser 1 e nao 2
    skip_connections = [skip]
    ##Build blocks
    for _ in range(k_layers):
        """
        The blocks are built as described in [1]:
        "...the dilation is doubled for every layer up to a limit and then repeated: e.g. 
                      1, 2, 4, ..., 512, 1, 2, 4, ..., 512, 1, 2, 4, ..., 512." (d(t) = d(t-1)*2)"""
        #compute dilation rate
        atrous_rate = atrous_rate * 2
        res, skip = wavenet_block(n_atrous_filters, atrous_filter_size, atrous_rate)(res)
        if atrous_rate % max_dil_rate == 0:
            atrous_rate = 1
        #accumulate block inputs
        skip_connections.append(skip)
    #parameterized skip connections, accumulated outputs 
    par_skip = Add()(skip_connections)
    par_skip = Activation('relu')(par_skip)
    par_skip = Conv1D(1, 1, activation='relu')(par_skip)
    par_skip = Conv1D(1, 1)(par_skip)
    #flatten output to feed to fc layer
    flat_par_skip = Flatten()(par_skip)
    #the output has the same dimension of the input
    out = Activation('softmax')(flat_par_skip)
    #end of skip parameterization ^
    #build model
    model = Model(inputs=input_tensor, outputs=out)
    ##Compile WaveNet
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    #verbose
    if summary:
        model.summary()
    
    return model


def dilated_gated_conv_res_net(input_dim, out_dim=2, k_layers=36, k_layers_dense=0, max_dil_rate=8, n_atrous_filters=64, atrous_filter_size=3,
                                n_units=64, loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'], summary=True):
    """
    Network architecture as decribed in [2]
    Very similar to WaveNet 
    The default parameters' values are the same as in [2] (except `optimizer` [2] uses Asynchronous SGD) 
    Main differences
        1. Just uses residual connections => WaveNet uses parameterized skip connection and residuals
        2. Use network's residual output => WaveNet discards last layer's output residual
        3. Before stacked blocks the Input is transformed using 1x1 conv. => WaveNet performs causal conv. before stack
        4. Dense layer computes p(speech), p(non speech), where the input is the model's accumulated output (3. ^)
    """
    #rate of dilation
    atrous_rate = 1
    #Input tensor
    input_tensor = Input(shape=(input_dim, 1)) 
    #(dim, match)
    #activation ??
    dim_match_in = Conv1D(1, 1, activation='linear')(input_tensor)
    ##First block 
    #non-accumulated input
    net_out, _ = wavenet_block(n_atrous_filters, atrous_filter_size, atrous_rate)(dim_match_in)
    ##Build blocks
    for _ in range(k_layers):
        """
        The blocks are built as described in [2]:
        dilation 1, 2, 4, 8, 1, 2, 4, 8, ...
        """
        #compute dilation rate
        atrous_rate = atrous_rate * 2
        #store for last layer
        res, _ = wavenet_block(n_atrous_filters, atrous_filter_size, atrous_rate)(net_out)
        #set dilation rate to 1
        if atrous_rate % max_dil_rate == 0:
            atrous_rate = 1
    #[2] only has output layer with softmax
    dense_in = res
    for _ in range(k_layers_dense):
        dense_in = Dense(n_units, kernel_initializer='glorot_uniform', activation='relu')(dense_in)
    #Output layer
    #flatten for output fc layer
    dense_in = Flatten()(dense_in)
    """
    In [2] the output has two targets: speech and non-speech
    """
    out = Dense(out_dim, kernel_initializer='glorot_uniform', activation='softmax')(dense_in)
    #build model
    model = Model(inputs=input_tensor, outputs=out)
    #compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    #verbose
    if summary:
        model.summary()

    return model 

# wavenet = dilated_gated_conv_res_net(256)