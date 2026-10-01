def neuron(x, w, b):
    return x * w + b

def loss(prediction, target):
    return (prediction- target) ** 2

x = 3
target = 6
b = 0
x = 3
w = 1
b = 0
target = 6

prediction = neuron(x, w, b)
current_loss = loss(prediction, target)

step = 0.1

new_w = w + step

new_prediction = neuron(x, new_w, b)
new_loss = loss(new_prediction, target)

print("New loss:", new_loss)

gradient = 2 * x * (prediction - target)

learning_rate = 0.1

w = w - learning_rate * gradient

print("New w:", w)

for step in range(10):
    prediction = neuron(x, w, b)
    current_loss = loss(prediction, target)

    gradient = 2 * x * (prediction - target)

    w = w - learning_rate * gradient

    print("Step:", step, "w:", w, "loss:", current_loss)