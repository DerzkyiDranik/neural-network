import numpy as np

from network import Network

train_data = [
    ([1, 1], [10, 12, 13]),
    ([2, 1], [12, 14, 16]),
    ([1, 2], [13, 15, 17]),
    ([3, 2], [17, 19, 20]),
]

test_data = [
    ([2, 2], [15, 17, 19])
]

def main():
    network = Network(
        layer_sizes=[2, 4, 3],
        learning_rate=0.001
    )
    
    network.train(train_data, epochs=5000)
    
    train_loss = network.evaluate(train_data)
    test_loss = network.evaluate(train_data)
    
    print(f"Ошибка на обучающих данных: {train_loss:.4f}")
    print(f"Ошибка на тестовых данных: {test_loss:.4f}")
    
    test_inputs = [
        [1, 1],
        [1, 2],
        [2, 1]
    ]
        
    network.save("model.npz")
    
    original_prediction = network.predict([2, 2])
    
    loaded_network = Network(
        layer_sizes=[2, 4, 3],
        learning_rate=0.001
    )
    
    loaded_network.load("model.npz")
    
    loaded_prediction = loaded_network.predict([2, 2])
    
    print("До загрузки:", original_prediction)
    print("После загрузки:", loaded_prediction)
    print("Предсказания совпадают:", np.allclose(original_prediction, loaded_prediction))
    
    # print("\nПредсказание:")
    
    # for inputs in test_inputs:
    #     prediction = network.predict(inputs)
        
    #     print(f"Вход: {inputs} -> Предсказание: {prediction}")
    
if __name__ == "__main__":
    main()
