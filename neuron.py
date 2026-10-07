def layer(x, weights, biases):
    return weights @ x + biases

data = [
    ([1, 1], [10, 12, 13]),
    ([2, 1], [12, 14, 16]),
    ([1, 2], [13, 15, 17]),
    ([3, 2], [17, 19, 20]),
    ([2, 2], [15, 17, 19])
]

import numpy as np

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

        predictions = layer(x, weights, biases)
        
        current_loss = np.sum((predictions - targets) ** 2)
        total_loss = total_loss + current_loss
        
        gradient_weights = 2 * np.outer(predictions - targets, x)
        gradient_biases = 2 * (predictions - targets)
        
        weights = weights - learning_rate * gradient_weights
        biases = biases - learning_rate * gradient_biases
