import numpy as np

def relu(x):
    return np.maximum(0, x)

def forward_layer(x, weights, biases):
    z = weights @ x + biases
    output = relu(z)
    
    return z, output

def create_layer(input_size, output_size):
    weights = np.random.randn(output_size, input_size) * 0.1
    biases = np.zeros(output_size)
    
    return weights, biases

data = [
    ([1, 1], [10, 12, 13]),
    ([2, 1], [12, 14, 16]),
    ([1, 2], [13, 15, 17]),
    ([3, 2], [17, 19, 20]),
    ([2, 2], [15, 17, 19])
]

weights = np.array([
    [1, 1],
    [1, 1],
    [1, 1]
], dtype=float)

biases = np.array([0, 0, 0], dtype=float)

learning_rate = 0.01

loss_history = []

for epoch in range(100):

    total_loss = 0

    for x, targets in data:
        x = np.array(x, dtype=float)
        targets = np.array(targets, dtype=float)
        
        z, predictions = forward_layer(x, weights, biases)
        
        erorrs = predictions - targets
        
        current_loss = np.sum(erorrs ** 2)
        total_loss += current_loss
        
        gradient = 2 * erorrs * (z > 0)
        
        gradient_weights = np.outer(gradient, x)
        gradient_biases = gradient
        
        weights = weights - learning_rate * gradient_weights
        biases = biases - learning_rate * gradient_biases
        
        average_loss = total_loss / (len(data) * 3)
        loss_history.append(average_loss)
        
        if epoch % 10 ==0:
            print("Epoch:", epoch)
            print("Average loss:", average_loss)
            
    print("initial:", loss_history[0])
    print("Final loss:", loss_history[-1])
    print("Weights:", weights)
    print("Biases:", biases)
