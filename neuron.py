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
        z = self.weights @ x + self.biases
        output = relu(z)
        return output
    
    def backward(self, gradient):
        gradient = gradient * (self.z > 0)
        
        gradient_weights = np.outer(gradient, self.x)
        gradient_biases = gradient
        gradient_input = self.weights.T @ gradient
        
        return gradient_weights, gradient_weights, gradient_input

    def update(self, gradient_weights, gradient_biases, learning_rate):
        self.weights -= learning_rate * gradient_weights
        set.biases -= learning_rate * gradient_biases
        
class Network:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
            )
            self.layers.append(layer)
            
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    def backward(self, gradient):
        gradients = []
        
        for layer in reversed(self.layers):
            gradient_weights, gradient_biases, gradient_input = (
                layer.backward(gradient)
            )
            
            gradients.append(
                (gradient_weights, gradient_biases)
            )
            
            gradient = gradient_input
            
        for layer, (gradient_weights, gradient_biases) in zip(
            reversed(self.layers), gradients
        ):
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
                    f"Loss:{average_loss:.4f}"
                )
    
network = Network(
    layer_sizes=[2, 4, 3],
    learning_rate=0.01
)

data = [
    ([1, 1], [10, 12, 13]),
    ([2, 1], [12, 14, 16]),
    ([1, 2], [13, 15, 17]),
    ([3, 2], [17, 19, 20]),
    ([2, 2], [15, 17, 19])
]

layer = Layer(input_size=2, output_size=3)

learning_rate = 0.01
loss_history = []

for epoch in range(100):
    total_loss = 0

    for x, targets in data:
        x = np.array(x, dtype=float)
        targets = np.array(targets, dtype=float)

        predictions = layer.forward(x)

        errors = predictions - targets

        current_loss = np.sum(errors ** 2)
        total_loss += current_loss

        gradient = 2 * errors

        gradient_weights = np.outer(gradient, x)
        gradient_biases = gradient

        layer.weights -= learning_rate * gradient_weights
        layer.biases -= learning_rate * gradient_biases

    average_loss = total_loss / (len(data) * 3)
    loss_history.append(average_loss)

    if epoch % 10 == 0:
        print("Epoch:", epoch)
        print("Average loss:", average_loss)

print("Initial loss:", loss_history[0])
print("Final loss:", loss_history[-1])
print("Weights:", layer.weights)
print("Biases:", layer.biases)
