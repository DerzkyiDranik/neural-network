from network import Network

data = [
    ([1, 1], [10, 12, 13]),
    ([2, 1], [12, 14, 16]),
    ([1, 2], [13, 15, 17]),
    ([3, 2], [17, 19, 20]),
    ([2, 2], [15, 17, 19]),
]

def main():
    network = Network(
        layer_sizes=[2, 4, 3],
        learning_rate=0.001
    )
    
    network.train(data, epochs=5000)
    
    loss = network.evaluate(data)
    print(f"\nИтоговый Loss: {loss:.4f}")
    
    test_inputs = [
        [1, 1],
        [1, 2],
        [2, 1]
    ]
    
    print("\nПредсказание:")
    
    for inputs in test_inputs:
        prediction = network.predict(inputs)
        
        print(f"Вход: {inputs} -> Предсказание: {prediction}")
    
if __name__ == "__main__":
    main()
