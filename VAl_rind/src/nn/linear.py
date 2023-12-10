import val_rind , Parameters , Module

import numpy as np

class linear(Module): 
    '''
    Linear layer for the Transformer model. Inherits from the Module class

    Parameters

        1) in_feats : Number of input features
        2) out_feats : Number of output features
    '''

    def __init__(self , in_feats , out_feats):
        '''
        Constructor for the linear class
        '''

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.in_col = np.random.uniform(0.01 , 0.001 , self.in_feats)
        self.out_col = np.random.uniform(0.01 , 0.001 , self.out_feats)


        self.params = [self.in_col , self.out_col] 
        
        self.params = [Parameters(val) for val in self.params]
        
    def forward(self , inps):
        '''
        Forward pass for the linear class

        Parameters

            1) inps : Inputs to the linear layer
        '''

        if len(inps.shape) == 1 : # 1D
            return_val = val_rind.matmul(self.in_col , inps , 
                                self.in_col , self.out_col)
        
        elif len(inps.shape) == 2 : # 2D 
            return_val = np.vstack([val_rind.matmul(self.in_col , val , 
                                           self.in_col , self.out_col)
                                    for val in inps])

        elif len(inps.shape) == 3 : # Batch
            return_val = []

            for batch in inps:

                batched = np.vstack([val_rind.matmul(self.in_col , val , 
                                            self.in_col , self.out_col)
                                     for val in batch])
            
                return_val.append(batched)

            return_val = np.stack(return_val)

        else : raise ValueError(f'Inputs of shape {inps.shape} cannot be sent to processed')

        return return_val
