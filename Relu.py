import numpy as np

class Relu:
    def forward(self, input_data):
        self.input_cache = input_data
        return np.maximum(0, input_data)        
    
    def backward(self, delta_current, learning_rate = -1): # Dummy learning rate
        grad = np.zeros_like(self.input_cache)
        grad[self.input_cache > 0] = 1
        return delta_current * grad