import numpy as np

class Dense:
    def  __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)       
        self.weights.reshape((input_size, -1))
        self.bias = np.zeros(output_size)
    
    def forward(self, input_data):
        self.input_cache = input_data
        Z = input_data.dot(self.weights) + self.bias
        return Z
    
    def backward(self, delta_current, learning_rate=0.005):
        # Calculate gradients for W and B
        self.grads = (self.input_cache.T).dot(delta_current)
        self.grad_bias = delta_current.sum(axis=0)
        

        delta_prev = delta_current.dot(self.weights.T)

        # Update Weights and bias
        self.weights -= learning_rate * self.grads
        self.bias -= learning_rate * self.grad_bias

        # Pass the error back to the previous layer
        return delta_prev