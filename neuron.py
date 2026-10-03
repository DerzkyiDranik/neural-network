def neuron(x, w, b):
    return x * w + b

def loss(prediction, target):
    return (prediction- target) ** 2

data = [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8)
]

w = 1
b = 0
learning_rate = 0.1

print(data)

for epoch in range(100):
    total_loss = 0

    for x, target in data:
        # обучение

        prediction = neuron(x, w, b)
        current_loss = loss(prediction, target)
        
        total_loss = total_loss + current_loss
        
        gradient = 2 * x * (prediction - target)

        w = w - learning_rate * gradient

        average_loss = total_loss / len(data)

        print("Epoch:", epoch, "Average loss:", average_loss)

        print("x:", x, "target:", target, "prediction:", prediction, "loss", current_loss)


