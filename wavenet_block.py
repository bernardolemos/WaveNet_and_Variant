"""
1. WaveNet: : A Generative Model for Raw Audio - https://arxiv.org/pdf/1609.03499.pdf
"""
from tensorflow.keras.layers import Add, Multiply, Conv1D

def wavenet_block(n_atrous_filters, atrous_filter_size, atrous_rate):
    """
    WaveNet block implementation as described in [1]
    """
    def build_wavenet_block(input_tensor):
        residual = input_tensor
        ## Casual dilated convolutions
        # Note: padding is set to 'causal', from Keras docs. => "causal" results in causal (dilated) convolutions, 
        # e.g. output[t] does not depend on input[t + 1:]. A zero padding is used such that the output has the same 
        # length as the original input. Useful when modeling temporal data where the model should not violate the temporal 
        # order. See WaveNet: A Generative Model for Raw Audio, section 2.1.
        """ 'causal' vs 'same' """
        """ ? 'channels_last' ? """
        tanh_out = Conv1D(n_atrous_filters, atrous_filter_size, padding='same', 
                            data_format='channels_last', dilation_rate=atrous_rate, activation='tanh')(input_tensor)
        sigmoid_out = Conv1D(n_atrous_filters, atrous_filter_size, padding='same', 
                            data_format='channels_last', dilation_rate=atrous_rate, activation='sigmoid')(input_tensor)
        #apply gated connection
        gated_out = Multiply()([tanh_out, sigmoid_out])
        ##1x1 convolution 
        #(dim.match) in [2] (similar to WaveNet)
        skip_out = Conv1D(1, 1, activation='linear')(gated_out) #to be accumulated
        ##Add block's input and output (residual)
        out = Add()([skip_out, residual])
        
        return out, skip_out
    
    return build_wavenet_block