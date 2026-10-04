def neuron(x, w, b):
    return x * w + b

def loss(prediction, target):
    return (prediction- target) ** 2

data = [
    (1, 10),
    (2, 12),
    (3, 14),
    (4, 16),
    (5, 18)
]

w = 1
b = 0
learning_rate = 0.1

print(data)

for epoch in range(100):
    print("Final weight:", w)

    x = 5

    prediction = neuron(x, w, b)

    print("Prediction", prediction)

    total_loss = 0

    for x, target in data:
        # обучение

        prediction = neuron(x, w, b)
        current_loss = loss(prediction, target)
        
        total_loss = total_loss + current_loss
        
        gradient_w = 2 * x * (prediction - target)

        gradient_b = 2 * (prediction - target)

        w = w - learning_rate * gradient_w

        b = b - learning_rate * gradient_b

    average_loss = total_loss / len(data)

    print("Epoch:", epoch, "Average loss:", average_loss)

    print("x:", x, "target:", target, "prediction:", prediction, "loss", current_loss)
print("Final weight:", w)
print("Final bias:", b)