def neuron(x, w, b):
    return x * w + b
def loss(prediction, target):
    return (prediction- target) ** 2

data = [
    (1, 10),
    (2, 13),
    (3, 13),
    (4, 17),
    (5, 19)
]

w = 1
b = 0
learning_rate = 0.01

print(data)

loss_history = []
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
        
        print("Before update:")
        print("Weight:", w)
        print("Bias:", b)

        w = w - learning_rate * gradient_w
        b = b - learning_rate * gradient_b
        
        print("After update:")
        print("Weight:", w)
        print("Bias:", b)

    average_loss = total_loss / len(data)
    loss_history.append(average_loss)

    print("Epoch:", epoch, "Average loss:", average_loss)
    print("Initial loss:", loss_history[0])
    print("Final loss:", loss_history[-1])