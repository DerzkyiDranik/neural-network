def neuron(x, w, b):
    return x * w + b

def loss(prediction, target):
    return (prediction- target) ** 2

x = 3
target = 6
b = 0

for w in [0, 1, 2, 3, 4]:
    prediction = neuron(x, w, b)
    error = loss(prediction,target)

    print("w", w, "prediction:", prediction, "loss", error)