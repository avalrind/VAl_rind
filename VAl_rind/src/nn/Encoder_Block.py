import val_rind , mhsa , linear , Module

class Encoder_Block(Module):
    '''
    Encoder Block class for the Transformer model. Inherits from the Module class

    Parameters
    
            1) model_dim : Model dimension
    '''

    def __init__(self , model_dim):
            
        '''
        Constructor for the Encoder Block class
        '''
        self.model_dim = model_dim
        
        self.mhsa = mhsa(self.model_dim)
        self.linear_ = linear(self.model_dim , self.model_dim)

        self.parameters = [self.mhsa.parameters]

    def forward(self , inps):

        '''
        Forward pass for the Encoder Block class

        Parameters
            
                1) inps : Inputs to the Encoder Block
        '''

        attention , weights = self.mhsa.forward(inps , inps, inps)

        attention = val_rind.layer_norm(attention)

        linear_attention = self.linear_.forward(attention)
        attention = val_rind.layer_norm(linear_attention + attention)

        return attention , weights
