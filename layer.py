import numpy as np

def relu(x): 
    return np.maximum(0, x)

class Layer: 
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        
        self.biases = np.zeros(output_size)
        
        self.x = None
        self.z = None

    def forward(self, x):
        self.x = x
        self.z = self.weights @ x + self.biases
        
        output = relu(self.z)
        
        return output
    
    def backward(self, gradient):
        gradient = gradient * (self.z > 0)
        
        gradient_weights = np.outer(
            gradient, self.x
        )
        
        gradient_biases = gradient
        gradient_input = self.weights.T @ gradient
        
        return (
            gradient_weights,
            gradient_biases,
            gradient_input
        )
        
    def update(
        self,
        gradient_weights,
        gradient_biases,
        learning_rate
    ):
        self.weights -= learning_rate * gradient_weights
        self.biases -= learning_rate * gradient_biases