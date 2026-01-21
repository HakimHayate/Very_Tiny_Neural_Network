import numpy as np

class Sigmoid:

    def sigmoid(self, input_data):
        return 1/(1+np.exp(-input_data))
    
    def forward(self, input_data):
        self.input_cache = input_data
        return self.sigmoid(input_data)   
    
    def backward(self, delta_current, learning_rate = -1): # Dummy learning rate
        s = self.sigmoid(self.input_cache)
        return delta_current * s * (1 - s)