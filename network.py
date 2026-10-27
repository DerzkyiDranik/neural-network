import numpy as np

from layer import Layer

def mse_loss(predictions, targets):
                errors = predictions - targets
                return np.mean(errors ** 2)

class Network:
    def __init__(self, layer_sizes, learning_rate=0.001):
        self.layers = []
        self.learning_rate = learning_rate
        
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],

            use_relu=(i < len(layer_sizes) - 2))
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

        return gradients
    
    def evaluate(self, data):
        total_loss = 0
        
        for inputs, targets in data:
            inputs = np.array(inputs, dtype=float)
            targets = np.array(targets, dtype=float)
            
            predictions = self.predict(inputs)
            total_loss += mse_loss(predictions, targets) * len(targets)
            
        return total_loss / sum(
            len(targets) for _, targets in data
        )
    
    def predict(self, x):
        x = np.array(x, dtype=float)
        return self.forward(x)
    
    def save(self, filepath="model.npz"):
        parameters = {}
        
        for i, layer in enumerate(self.layers):
            parameters[f"weights_{i}"] = layer.weights
            parameters[f"biases_{i}"] = layer.biases
                
        np.savez(filepath, **parameters)
        
    def load(self, filepath="model.npz"):
        with np.load(filepath) as parameters:
        
            for i, layer in enumerate(self.layers):
                layer.weights = parameters[f"weights_{i}"].copy()
                layer.biases = parameters[f"biases_{i}"].copy()
                
                parameters.close
    
    def train(self, data, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            gradients = []

            for inputs, targets in data:
                inputs = np.array(inputs, dtype=float)
                targets = np.array(targets, dtype=float)

                predictions = self.forward(inputs)
                
                errors = predictions - targets
                total_loss += mse_loss(predictions, targets) * len(targets)

                gradient = 2 * errors
                sample_gradients = self.backward(gradient)

                gradients.append(sample_gradients)

            for layer_index, layer in enumerate(self.layers):
                gradient_weights = np.mean(
                    [
                        sample_gradients[
                            len(self.layers) - 1 - layer_index
                        ][0]
                        for sample_gradients in gradients
                    ],
                    axis=0
                )

                gradient_biases = np.mean(
                    [
                        sample_gradients[
                            len(self.layers) - 1 - layer_index
                        ][1]
                        for sample_gradients in gradients
                    ],
                    axis=0
                )

                layer.update(
                    gradient_weights,
                    gradient_biases,
                    self.learning_rate
                )

            average_loss = total_loss / (
                len(data) * len(data[0][1])
            )
            
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}, "
                    f"Loss: {average_loss:.4f}"
                )
