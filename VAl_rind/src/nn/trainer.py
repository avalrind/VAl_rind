from nn import Module

class gnn_model(Module) : 
    '''
    Trainer Class for Graph Neural Network model

    Parameters

        1) layers : Layers of the model
    '''

    def __init__(self , layers) : 
        '''
        Constructor for the gnn_model class
        '''

        self.layer = layers

    def forward(self , inps) :
        '''
        Forward pass for the gnn_model class

        Parameters

            1) inps : Inputs to the model
        '''

        for layer in self.layers : 

            value = layer.value

            if layer.is_inp_node and inps[1] == layer : 

                value += inps 
                value *= layer.weight
                value += layer.bias

            connections = layer.connections 

            value += sum(
                [val.value 
                 for val 
                 in connections]
            )

            value += layer.weight
            value += layer.bias

            if layer.is_out_node : yield layer.value
