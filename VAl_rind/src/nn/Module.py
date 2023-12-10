import Parameters

class Module : 
    '''
    Module class for the Transformer model. Inherits from the Parameters class
    '''

    def __init__(self) : 
        '''
        Constructor for the Module class
        '''
        
        self._parameters = {}

    def _register_parameter(self , name , param) : 
        '''
        Registers a parameter to the Module class

        Parameters

            1) name : Name of the parameter
            2) param : Parameter to be registered
        '''
        
        self._parameters[name] = param

    def _register_submodule(self , name , module) :
        '''
        Registers a submodule to the Module class

        Parameters

            1) name : Name of the submodule
            2) module : Submodule to be registered
        ''' 

        for sub_name , sub_param in module._parameters.items() :

            self._register_parameter(f'{name}.{sub_name}' , sub_param)

    def parameters(self) : 
        '''
        Returns the parameters of the Module class
        '''
        
        return self._parameters.values()

    def __setattr__(self , name , value) : 
        '''
        Sets the value of the attribute

        Parameters

            1) name : Name of the attribute
            2) value : Value of the attribute
        '''

        if isinstance(value , Parameters) : self._register_parameter(name , value)
        elif isinstance(value , Module) : self._register_submodule(name , value)

        super().__setattr__(name , value)
