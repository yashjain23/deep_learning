
from nn_train import  forward_propagation
import numpy as np
import pickle

def load_parameters(save_path="parameters.pkl"):
    with open(save_path, "rb") as file:
        parameters = pickle.load(file)
    print(f"Parameters loaded from {save_path}")
    return parameters

def test_model(X, Y, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = (AL > 0.5).astype(int)
    accuracy = np.mean(predictions == Y) * 100
    print(f"Accuracy on test data: {accuracy:.2f}%")
    return predictions

if __name__ == "__main__":
    test_data = np.loadtxt("mnist_test.csv", dtype=int, delimiter=',', skiprows=1)
    test_X = test_data[:, 1:].T
    test_output = test_data[:, 0]
    test_Y0 = np.zeros((len(test_output), 10), dtype=int)
    test_Y0[np.arange(len(test_output)), test_output] = 1
    test_Y0 = test_Y0.T

    loaded_parameters = load_parameters()
    test_model(test_X, test_Y0, loaded_parameters)