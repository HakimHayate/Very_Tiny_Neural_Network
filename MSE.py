import numpy as np

class MSE:
    def calculate(self, predict, target):
        mse =  ((predict - target)**2).sum() / len(target)
        # print(f'Current error = {mse}')
        return mse
    
    def backward(self, predict, target):
        grad = 2 * (predict - target) / len(target)
        return grad
    
