import pickle
import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


# Initialize parameters
def initialize_parameters(layer_dims):
    np.random.seed(0)
    parameters = {}
    L = len(layer_dims)  # Number of layers in the network

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return parameters


# Forward propagation
def forward_propagation(X, parameters):
    caches = {}
    A = X
    L = len(parameters) // 2  # Number of layers

    caches[f"A{0}"] = X
    for l in range(1, L + 1):
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        Z = np.dot(W, A) + b
        A = relu(Z) if l < L else sigmoid(Z)
        caches[f"Z{l}"] = Z
        caches[f"A{l}"] = A  # Store inputs of this layer

    return A, caches


# Backward propagation
def backward_propagation(X, Y, parameters, caches):
    grads = {}
    L = len(parameters) // 2
    m = X.shape[1]
    Y = Y.reshape(caches[f"A{L}"].shape)

    # Derivative of the loss with respect to the output
    dA = -(np.divide(Y, caches[f"A{L}"]) - np.divide(1 - Y, 1 - caches[f"A{L}"]))

    for l in reversed(range(1, L + 1)):
        Z = caches[f"Z{l}"]
        A_prev = caches[f"A{l - 1}"]
        W = parameters[f"W{l}"]

        dZ = dA * (sigmoid_derivative(Z) if l == L else relu_derivative(Z))
        grads[f"dW{l}"] = np.dot(dZ, A_prev.T) / m
        grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(W.T, dZ)

    return grads


# Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters


# Train the model
def train_neural_network(X, Y, layer_dims, learning_rate=0.01, num_iterations=1000, save_path="parameters.pkl"):
    parameters = initialize_parameters(layer_dims)

    costs = []

    # Initialize live plot
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Cost vs Iteration")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, num_iterations)
    ax.set_ylim(0, 1)

    for i in range(num_iterations):
        # Forward propagation
        AL, caches = forward_propagation(X, parameters)

        # Compute cost (Binary cross-entropy)
        cost = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        costs.append(cost)
        # Backward propagation
        grads = backward_propagation(X, Y, parameters, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print and plot cost every 10 iterations
        if i%10==0:
            line.set_xdata(range(len(costs)))
            line.set_ydata(costs)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
            print(f"Cost after iteration {i}: {cost}")

    with open(save_path, "wb") as file:
        pickle.dump(parameters, file)
    print(f"Parameters saved to {save_path}")

    plt.ioff()
    plt.show()

    return parameters


if __name__ == "__main__":

    # Input and Output
    data = np.loadtxt("mnist_train.csv", dtype=int, delimiter=',', skiprows=1)
    A0 = data[:, 1:].T
    output = data[:, 0]
    Y0 = np.zeros((len(output), 10), dtype=int)
    Y0[np.arange(len(output)), output] = 1
    Y0 = Y0.T

    layer_dims = [784, 512, 256, 128, 64, 32, 16, 10]  # Example: 7-layer network
    trained_parameters = train_neural_network(A0, Y0, layer_dims, learning_rate=0.001, num_iterations=10000)




