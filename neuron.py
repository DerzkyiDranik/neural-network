import numpy as np

def relu(x):
    return np.maximum(0, x)

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.zeros(output_size)

    def forward(self, x):
        z = self.weights @ x + self.biases
        output = relu(z)
        return z, output

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

        z, predictions = layer.forward(x)

        errors = predictions - targets

        current_loss = np.sum(errors ** 2)
        total_loss += current_loss

        gradient = 2 * errors * (z > 0)

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
