import numpy as np

from layer import Layer

class Network:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
input_size=layer_sizes[i],
output_size=layer_sizes[i + 1]
            )

            self.layers.append(layer)
            
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    def backward(self, gradient):
        gradients = []
        
        for layer in reversed(self.layers):
            (
                gradient_weights,
                gradient_biases,
                gradient_input
            ) = layer.backward(gradient)

            gradients.append(
                (gradient_weights, gradient_biases)
            )
            
            gradient = gradient_input
            
        for layer, (
            gradient_weights,
            gradient_biases
        ) in zip(reversed(self.layers), gradients):
            layer.update(
                gradient_weights,
                gradient_biases,
                self.learning_rate
            )           
    
    def predict(self, x):
        x = np.array(x, dtype=float)
        return self.forward(x)
    
    def train(self, data, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            
            for inputs, targets in data:
                inputs = np.array(inputs, dtype=float)
                targets = np.array(targets, dtype=float)
                
                predictions = self.forward(inputs)
                
                errors = predictions - targets
                
                current_loss = np.sum(errors ** 2)
                total_loss += current_loss
                
                gradient = 2 * errors
                self.backward(gradient)
            
            average_loss = total_loss / (len(data) * len(targets))
            
            if ( epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1},"
                    f"Loss: {average_loss:.4f}"
                    )
