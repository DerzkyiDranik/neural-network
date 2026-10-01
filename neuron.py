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
lerning_rate = 0.1

for x, target in data:
    prediction = neuron(x, w, b)
    current_loss = loss(prediction, target)

    gradient = 2 * x * (prediction - target)

    w = w - lerning_rate * gradient

    print("x:", x, "target:", target, "prediction:", prediction, "loss", current_loss)
