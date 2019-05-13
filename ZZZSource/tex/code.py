from keras import backend as K
from keras.layers import Layer

#Marginal Layer Class for Keras
class MarginalLayer(Layer):
    
    #Saving input variables and initializing layer
    def __init__(self, hidden_units=12, **kwargs):
        self.hidden_units = hidden_units
        super(MarginalLayer, self).__init__(**kwargs)
        
    #Builds the marginal layer
    def build(self, input_shape):
        
        #Initializers
        winit = keras.initializers.RandomNormal(mean=1.0, stddev=0.05, seed=None)
        binit = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        oinit = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
                                
        #Creating input weight matrix
        self.win = self.add_weight(name = 'marginal_input_weight',
                                   shape = (input_shape[1], self.hidden_units),
                                   initializer = winit)
        
        #Creating the input bias matrix
        self.bin = self.add_weight(name = 'marginal_input_bias',
                                   shape = (input_shape[1], self.hidden_units),
                                   initializer = binit)
        
        #Creating the output weight matrix
        self.wout = self.add_weight(name = 'marginal_output_weight',
                                   shape = (input_shape[1], self.hidden_units),
                                   initializer = oinit)
        
        #Creating the output bias matrix
        self.bout = self.add_weight(name = 'marginal_output_bias',
                                   shape = (input_shape[1], self.hidden_units),
                                   initializer = binit)
        
        #Building layer
        super(MarginalLayer, self).build(input_shape) 
          
    #Calls the layer on the input
    def call(self, x):
                
        #Marginal calculations
        a = K.expand_dims(x, axis=2)
        
        #Expanding win along axis 0 and multiplying it with variable a and adding a bias
        b = a * K.expand_dims(self.win, axis=0) + K.expand_dims(self.bin, axis=0)
        
        #Passing b through an activation, multiplying by wout, and adding a bias
        c = K.sigmoid(b) * K.expand_dims(self.wout, axis=0) + \
            K.expand_dims(self.bout, axis=0)
        
        #Adding lienar term back into the marginal
        c = K.concatenate([c, a], axis=2)
        
        #Summing along the second axis
        d = K.sum(c, axis=2, keepdims=False)
        
        #Returning marginal output
        return d

    #Computing the output shape
    def compute_output_shape(self, input_shape):
        return input_shape