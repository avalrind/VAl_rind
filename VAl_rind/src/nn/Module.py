import Parameters

class Module : 

    def __init__(self) : self._parameters = {}

    def _register_parameter(self , name , param) : self._parameters[name] = param

    def _register_submodule(self , name , module) : 

        for sub_name , sub_param in module._parameters.items() :

            self._register_parameter(f'{name}.{sub_name}' , sub_param)

    def parameters(self) : return self._parameters.values()

    def __setattr__(self , name , value) : 

        if isinstance(value , Parameters) : self._register_parameter(name , value)
        elif isinstance(value , Module) : self._register_submodule(name , value)

        super().__setattr__(name , value)