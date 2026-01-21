import MSE
import numpy as np

class Model:
    def __init__(self, layers, loss_function=None):
        self.layers = layers
        if loss_function == None:
            self.loss_function = MSE.MSE()
        else:
            self.loss_function = loss_function
    
    def forward(self, input_data):
        tmp = input_data
        for layer in self.layers:
            tmp = layer.forward(tmp)
        return tmp
    
    def backward(self, prediction, target, learning_rate):
        delta = self.loss_function.backward(prediction, target)

        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)
    
    def calculate_accuracy(self, pred, y):
        p = np.argmax(pred, axis=1)
        tmp = np.argmax(y, axis=1)
        return np.mean(p == tmp)
    
    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate=0.01):
        for _ in range(epochs):
            total_loss = 0
            for i in range(0, len(x_train)-batch_size, batch_size):
                x = x_train[i:i+batch_size]
                y = y_train[i:i+batch_size]

                pred = self.forward(x)

                total_loss += self.loss_function.calculate(pred, y)
                self.backward(pred, y, learning_rate)


            print(f'epoch {_} Train error = {total_loss} Test accuracy {self.calculate_accuracy(self.forward(x_test), y_test)}')

    def predict(self, x):
        return self.forward(x)

    def summary(self):
        print("--- Résumé du Modèle ---")
        total_params = 0
        total_memory_bytes = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            params = 0
            memory = 0
            
            # Poids
            if hasattr(layer, 'weights'):
                params += layer.weights.size
                memory += layer.weights.nbytes # Taille réelle en octets
            
            # Biais
            if hasattr(layer, 'bias'):
                params += layer.bias.size
                memory += layer.bias.nbytes
                
            print(f"Couche {i} ({layer_name}) : {params} params")
            total_params += params
            total_memory_bytes += memory
            
        print("------------------------")
        
        # Conversions
        size_kb = total_memory_bytes / 1024
        size_mb = size_kb / 1024
        
        print(f"Total Paramètres : {total_params}")
        print(f"Taille Mémoire   : {size_kb:.2f} KB ({size_mb:.4f} MB)")