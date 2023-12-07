class gnn_model : 

    def __init__(self , layers) : 

        self.layer = layers

    def forward(self , inps) :

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