import scipy
import numpy as np 
import linear , Module

class mhsa(Module) :
    '''
    Multi-Head Self Attention class for the Transformer model. Inherits from the Module class

    Parameters

        1) vocab_size : Size of the vocabulary
    '''

    def __init__(self , vocab_size):
        '''
        Constructor for the mhsa class
        '''

        self.vocab_size = vocab_size

        self.queries = linear(self.vocab_size , self.vocab_size)
        self.keys = linear(self.vocab_size , self.vocab_size)
        self.values = linear(self.vocab_size , self.vocab_size)

        self.parameters = [[self.queries.in_col , self.queries.out_col] , 
                           [self.keys.in_col , self.keys.out_col] , 
                           [self.values.in_col , self.values.out_col]]
        
    def forward(self , query , key , value , mask = None):
        '''
        Forward pass for the mhsa class

        Parameters

            1) query : Query for the mhsa
            2) key : Key for the mhsa
            3) value : Value for the mhsa
        '''

        query_output = self.queries.forward(query)
        key_output = self.keys.forward(key)
        value_output = self.values.forward(value)

        attention = (query_output * key_output) / (key_output.shape[-1] ** (1/2))

        if mask : attention = np.tril(attention)

        weights = scipy.special.softmax(attention , axis = 1)

        output = weights * value_output

        return output , weights
