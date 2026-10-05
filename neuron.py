def neuron(x1, x2, w1, w2, b):
    return x1 * w1 + x2 * w2 + b

def loss(prediction, target):
    return (prediction - target) ** 2

def layer(x1, x2, weights, biases):
    neuron1 = neuron(x1, x2, weights[0][0], weights[0][1], biases[0])
    neuron2 = neuron(x1, x2, weights[1][0], weights[1][1], biases[1])
    neuron3 = neuron(x1, x2, weights[2][0], weights[2][1], biases[2])

    return neuron1, neuron2, neuron3

data = [
    ([1, 1], [10, 12, 13]),
    ([2, 1], [12, 14, 16]),
    ([1, 2], [13, 15, 17]),
    ([3, 2], [17, 19, 20]),
    ([2, 2], [15, 17, 19])
]

weights = [
    [1, 1],
    [1, 1],
    [1, 1]
]

biases = [
    0,
    0,
    0
]

learning_rate = 0.01

loss_history = []

for epoch in range(100):

    total_loss = 0

    for x, targets in data:
        x1 = x[0]
        x2 = x[1]

        predictions = layer(x1, x2, weights, biases)
        
        for i in range(3):
            
            prediction = predictions[i]
            target = targets[i]
        
            current_loss = loss(prediction, target)
            
            total_loss = total_loss + current_loss
            
            gradient_w1 = 2 * x1 * (prediction - target)
            gradient_w2 = 2 * x2 * (prediction - target)
            gradient_b = 2 * (prediction - target)
            
            weights[i][0] = weights[i][0]  - learning_rate * gradient_w1
            weights[i][1]  = weights[i][1] - learning_rate * gradient_w2
            biases[i] = biases[i] - learning_rate * gradient_b

    average_loss = total_loss / (len(data) * 3)
    loss_history.append(average_loss)
    
    if epoch % 10 == 0:
        print("Epoch:", epoch)
        print("Weights:", weights)
        print("Bias:", biases)
        print("Average loss:", average_loss)

print("Initial loss:", loss_history[0])
print("Final loss:", loss_history[-1])
print("Test:", layer(5, 3, weights, biases))