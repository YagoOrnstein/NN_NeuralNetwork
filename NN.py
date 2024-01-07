import sys
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

class Activation_fcn:

    def __init__(self):
        self.functions = {
            'linear': self.linear,
            'sigmoid': self.logistic,
            'logistic': self.logistic,
            'tanh': self.tanh,
            'relu': self.relu,
            'leaky_relu': self.leaky_relu,
            'swish': self.swish
        }

    # Calculate neuron output
    def output(self, layer, name, derivative=False):
        if name in self.functions:
            return self.functions[name](layer, derivative)
        else:
            sys.exit(f"Error: Activation function '{name}' not found.")

    # Identity activation function
    def linear(self, layer, derivative=False):
        return layer['activation_potential'] if not derivative else np.ones_like(layer['activation_potential'])

    # Logistic (sigmoid) activation function
    def logistic(self, layer, derivative=False):
        if not derivative:
            return 1.0 / (1.0 + np.exp(-layer['activation_potential']))
        else:
            return layer['output'] * (1.0 - layer['output'])

    # Hyperbolic tangent activation function
    def tanh(self, layer, derivative=False):
        if not derivative:
            exp_sum = np.exp(layer['activation_potential']) + np.exp(-layer['activation_potential'])
            return (np.exp(layer['activation_potential']) - np.exp(-layer['activation_potential'])) / exp_sum
        else:
            return 1.0 - np.power(layer['output'], 2)

    # ReLU activation function
    def relu(self, layer, derivative=False):
        return np.maximum(0, layer['activation_potential']) if not derivative else (layer['activation_potential'] >= 0)

    # Leaky ReLU activation function
    def leaky_relu(self, layer, derivative=False, alpha=0.01):
        if not derivative:
            return np.maximum(alpha * layer['activation_potential'], layer['activation_potential'])
        else:
            return np.where(layer['activation_potential'] > 0, 1, alpha)

    # Swish activation function
    def swish(self, layer, derivative=False):
        if not derivative:
            return layer['activation_potential'] / (1 + np.exp(-layer['activation_potential']))
        else:
            exp_term = np.exp(-layer['activation_potential'])
            return (1 + layer['activation_potential'] * (1 - exp_term) - exp_term) / np.power(1 + exp_term, 2)


# Loss function class
class Loss_fcn:

    def __init__(self):
        self.functions = {
            'mse': self.mse,
            'binary_cross_entropy': self.binary_cross_entropy
        }

    # Loss/error value calculated for all input data sample
    def output(self, name, expected, outputs, derivative):
        if name in self.functions:
            return self.functions[name](expected, outputs, derivative)
        else:
            sys.exit(f"Error: Loss function '{name}' not found.")

    # Mean Square Error loss function
    def mse(self, expected, outputs, derivative=False):
        if not derivative:
            return 0.5 * np.power(expected - outputs, 2)
        else:
            return -(expected - outputs)

    # Cross-entropy loss function
    def binary_cross_entropy(self, expected, outputs, derivative=False):
        if not derivative:
            return -expected * np.log(outputs) - (1 - expected) * np.log(1 - outputs)
        else:
            return -(expected / outputs - (1 - expected) / (1 - outputs))


# Initialize a network
class Neural_network(object):

    def __init__(self, structure=None):
        self.af = Activation_fcn()
        self.loss = Loss_fcn()
        if structure:
            self.create_network(structure)

    def create_network(self, structure):
        self.nnetwork = [structure[0]]
        for i in range(1, len(structure)):
            new_layer = {
                'weights': np.random.randn(structure[i]['units'], structure[i - 1]['units'] + structure[i]['bias']),
                'bias': structure[i]['bias'],
                'activation_function': structure[i]['activation_function'],
                'activation_potential': None,
                'delta': None,
                'output': None}
            self.nnetwork.append(new_layer)
        return self.nnetwork

    def forward_propagate(self, nnetwork, inputs):
        inp = inputs.copy()
        for i in range(1, len(nnetwork)):
            if nnetwork[i]['bias']:
                inp = np.append(inp, 1)
            nnetwork[i]['activation_potential'] = np.matmul(nnetwork[i]['weights'], inp).flatten()
            nnetwork[i]['output'] = self.af.output(nnetwork[i], nnetwork[i]['activation_function'], derivative=False)
            inp = nnetwork[i]['output']
        return inp

    def backward_propagate(self, loss_function, nnetwork, expected):
        N = len(nnetwork) - 1
        for i in range(N, 0, -1):
            errors = []
            if i < N:
                weights = nnetwork[i + 1]['weights']
                if nnetwork[i + 1]['bias']:
                    weights = weights[:, :-1]
                errors = np.matmul(nnetwork[i + 1]['delta'], weights)
            else:
                errors = self.loss.output(loss_function, expected, nnetwork[-1]['output'], derivative=True)

            nnetwork[i]['delta'] = np.multiply(errors, self.af.output(nnetwork[i], nnetwork[i]['activation_function'],
                                                                      derivative=True))

    def update_weights(self, nnetwork, inputs, l_rate):
        inp = inputs
        for i in range(1, len(nnetwork)):
            if nnetwork[i]['bias']:
                inp = np.append(inp, 1)
            nnetwork[i]['weights'] -= l_rate * np.matmul(nnetwork[i]['delta'].reshape(-1, 1), inp.reshape(1, -1))
            inp = nnetwork[i]['output']

    def train(self, nnetwork, x_train, y_train, l_rate=0.01, n_epoch=100, loss_function='mse', verbose=1):
        history = []
        for epoch in range(n_epoch):
            sum_error = 0
            for iter, (x_row, y_row) in enumerate(zip(x_train, y_train)):
                self.forward_propagate(nnetwork, x_row)
                error = np.sum(self.loss.output(loss_function, y_row, nnetwork[-1]['output'], derivative=False))
                sum_error += error
                self.backward_propagate(loss_function, nnetwork, y_row)
                self.update_weights(nnetwork, x_row, l_rate

                                    )

            history.append(sum_error)

            if verbose > 0:
                print('>epoch=%d, loss=%.3f' % (epoch + 1, error))
        print('Results: epoch=%d, loss=%.3f' % (epoch + 1, sum_error))
        return history

    def predict(self, nnetwork, inputs):
        out = []
        for input in inputs:
            out.append(self.forward_propagate(nnetwork, input))
        return out


def generate_regression_data(n=30):
    X = np.linspace(-5, 5, n).reshape(-1, 1)
    y = np.sin(2 * X) + np.cos(X) + 5
    data_noise = np.random.normal(0, 0.2, n).reshape(-1, 1)
    Y = y + data_noise
    return X.reshape(-1, 1), Y.reshape(-1, 1)


def test_regression():
    X, Y = generate_regression_data()
    model = Neural_network()
    structure = [{'type': 'input', 'units': 1},
                 {'type': 'dense', 'units': 8, 'activation_function': 'tanh', 'bias': True},
                 {'type': 'dense', 'units': 8, 'activation_function': 'tanh', 'bias': True},
                 {'type': 'dense', 'units': 1, 'activation_function': 'linear', 'bias': True}]
    network = model.create_network(structure)
    history = model.train(network, X, Y, 0.01, 4000, 'mse', 0)
    plt.figure()
    plt.plot(history, label="Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
    predicted = model.predict(network, X)
    std = np.std(predicted - Y)
    print("\nStandard deviation = {}".format(std))
    X_test = np.linspace(-7, 7, 100).reshape(-1, 1)
    X_test = np.array(X_test).tolist()
    predicted = model.predict(network, X_test)
    plt.plot(X, Y, 'r--o', label="Training data")
    plt.plot(X_test, predicted, 'b--x', label="Predicted")
    plt.legend()
    plt.grid()
    plt.show()


def generate_classification_data(n=30):
    X1_1 = 1 + 4 * np.random.rand(n, 1)
    X1_2 = 1 + 4 * np.random.rand(n, 1)
    class1 = np.concatenate((X1_1, X1_2), axis=1)
    Y1 = np.ones(n)

    X0_1 = 3 + 4 * np.random.rand(n, 1)
    X0_2 = 3 + 4 * np.random.rand(n, 1)
    class0 = np.concatenate((X0_1, X0_2), axis=1)
    Y0 = np.zeros(n)

    X = np.concatenate((class1, class0))
    Y = np.concatenate((Y1, Y0))
    idx0 = [i for i, v in enumerate(Y) if v == 0]
    idx1 = [i for i, v in enumerate(Y) if v == 1]
    return X, Y, idx0, idx1


def test_classification():
    X, Y, idx0, idx1 = generate_classification_data()
    model = Neural_network()
    structure = [{'type': 'input', 'units': 2},
                 {'type': 'dense', 'units': 4, 'activation_function': 'leaky_relu', 'bias': True},
                 {'type': 'dense', 'units': 4, 'activation_function': 'swish', 'bias': True},
                 {'type': 'dense', 'units': 1, 'activation_function': 'logistic', 'bias': True}]
    network = model.create_network(structure)
    history = model.train(network, X, Y, 0.0001, 1000, 'binary_cross_entropy', 1)
    plt.figure()
    plt.plot(history, label="Train loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
    y = model.predict(network, X)
    t = 0
    for n, m in zip(y, Y):
        t += 1 - np.abs(np.round(np.array(n)) - np.array(m))
        print(f"pred = {n}, pred_round = {np.round(n)}, true = {m}")

    ACC = t / len(X)
    print(f"\nClassification accuracy = {ACC * 100}%")

    xx, yy = np.meshgrid(np.arange(0, 8, 0.1),
                         np.arange(0, 8, 0.1))
    X_vis = np.c_[xx.ravel(), yy.ravel()]
    h = model.predict(network, X_vis)
    h = np.array(h) >= 0.5
    h = np.reshape(h, (len(xx), len(yy)))

    plt.contourf(xx, yy, h, cmap='jet')
    plt.scatter(X[idx1, 0], X[idx1, 1], marker='^', c="red", edgecolors="white", label="class 1")
    plt.scatter(X[idx0, 0], X[idx0, 1], marker='o', c="blue", edgecolors="white", label="class 0")
    plt.show()


generate_classification_data()
test_classification()

generate_regression_data(30)
test_regression()
