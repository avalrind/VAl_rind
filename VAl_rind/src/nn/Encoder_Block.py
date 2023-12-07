import val_rind , mhsa , linear , Module

class Encoder_Block(Module):

    def __init__(self , model_dim):

        self.model_dim = model_dim
        
        self.mhsa = mhsa(self.model_dim)
        self.linear_ = linear(self.model_dim , self.model_dim)

        self.parameters = [self.mhsa.parameters]

    def forward(self , inps):

        attention , weights = self.mhsa.forward(inps , inps, inps)

        attention = val_rind.layer_norm(attention)

        linear_attention = self.linear_.forward(attention)
        attention = val_rind.layer_norm(linear_attention + attention)

        return attention , weights