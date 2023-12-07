import embedding , Encoder_Block , val_rind , Module

import numpy as np 

class Encoder (Module):

    def __init__(self , num_blocks , vocab_size , max_seq_len , model_dim):

        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim

        self.tok_embed = embedding(self.vocab_size , self.model_dim)
        self.pos_embed = embedding(self.vocab_size , self.model_dim)

        self.blokcs = []
        self.parameters = [self.tok_embed.parameters , 
                           self.pos_embed.parameters]
        
        for _ in range(self.num_blocks):

            obj = Encoder_Block(self.model_dim)

            self.blokcs.append(obj)
            self.parameters.append(obj.parameters)

    def forward(self , inps , mask = None):

        if len(inps.shape) == 1: 

            inputs = val_rind.padding(inps , self.max_seq_len)
            pos_vals = np.array([val for val in range(self.max_seq_len)])

            tok_embeds = self.tok_embed.forward(inputs)
            pos_embeds = self.pos_embed.forward(pos_vals)

            embeds = tok_embeds + pos_embeds

            for block in self.blokcs :
            
                embeds , weights = block.forward(embeds)

                return embeds , weights
        
        elif len(inps.shape) == 2:

            inputs = np.empty(shape = (inps.shape[0] , self.max_seq_len))

            for index in range(inps.shape[0]):

                inputs[index] = val_rind.padding(inps[index] , self.max_seq_len)

            index_vals = np.array([[val for val in range(self.max_seq_len)]
                                   for _ in range(inps.shape[0])])
            
            tok_embeds = self.tok_embed.forward(inputs)
            pos_embeds = self.pos_embed.forward(index_vals)

            embeds = tok_embeds + pos_embeds

            for block in self.blocks:

                embeds , weights = block.forward(embeds)

            return embeds , weights
        
        elif len(inps.shape) == 3 : 

            r_embeds = []
            r_weights = []

            for batch in inps:

                inputs = np.empty(shape = (batch.shape[0] , self.max_seq_len))

                for index in range(inps.shape[0]):

                    inputs[index] = val_rind.padding(batch[index] , self.max_seq_len)

                index_vals = np.array([[val for val in range(self.max_seq_len)]
                                       for _ in range(batch.shape[0])])
                
                tok_embeds = self.tok_embed.forward(inputs)
                pos_embeds = self.pos_embed.forward(index_vals)

                embeds = tok_embeds + pos_embeds

                for block in self.blocks:

                    embeds , weights = block.forward(embeds)

                r_embeds.append(embeds)
                r_weights.append(weights)

            r_embeds = np.stack(r_embeds)
            r_weights = np.stack(r_weights)

            return r_embeds , r_weights

        else : raise ValueError(f'Cannot Porcess inputs of shape {inps.shape}')