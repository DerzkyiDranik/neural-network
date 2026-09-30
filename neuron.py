def neuron(x, w, b):
    return x * w + b

def loss(prediction, target):
    return (prediction- target) ** 2

prediction = neuron(3, 1, 0)
target = 6

print(prediction)
print(loss(prediction, target))