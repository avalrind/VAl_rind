import Module , Parameters
import numpy as np

class embedding(Module): 

    def __init__(self , in_feats , out_feats): 
        
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.feats = np.random.rand(self.in_feats , self.out_feats)
        self.parameters = Parameters(self.feats)

    def forward(self , inps):

        if len(inps.shape) == 1 : return_val = np.vstack([
            self.feats[val] for val in inps])
        
        elif len(inps.shape) == 2 : return_val = np.stack(
            [np.vstack([
                self.feats[value] for value in val
            ]) for val in inps])

        elif len(inps.shape) == 3 : 

            return_val = []

            for batch in inps:

                batched = [[self.feats[value] for value in val]
                           for val in batch]
                
                return_val.append(batched)

            return_val = np.stack(return_val)

        else : raise ValueError(f'Cannot process inputs with shape{inps.shape}')

        return return_val
