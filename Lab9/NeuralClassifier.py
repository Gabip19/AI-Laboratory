import numpy as np


class NeuralClassifier:

    def __init__(self, hidden_layer_size=12, learning_rate=0.001, max_iterations=4000):
        self.__weights = []
        self.__bias = []
        self.__hidden_layer_size = hidden_layer_size
        self.__learning_rate = learning_rate
        self.__max_iterations = max_iterations

    @staticmethod
    def __softmax(x):
        exp_vector = np.exp(x)
        return exp_vector / exp_vector.sum(axis=1, keepdims=True)

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_prime(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def fit(self, x, y):
        no_features = len(x[0])
        no_outputs = len(set(y))

        y_with_classes = np.zeros((len(y), no_outputs))
        for i in range(len(y)):
            y_with_classes[i, y[i]] = 1
        y = y_with_classes

        # input -> hidden
        weight_in_to_hid = np.random.rand(no_features, self.__hidden_layer_size)
        bias_hid = np.random.randn(self.__hidden_layer_size)  # bias
        # hidden -> output
        weight_hid_to_out = np.random.rand(self.__hidden_layer_size, no_outputs)
        bias_out = np.random.randn(no_outputs)  # bias

        for epoch in range(self.__max_iterations):
            # start forward propagation
            y_hid = np.dot(x, weight_in_to_hid) + bias_hid
            y_hid_sigmoid = self.__sigmoid(y_hid)
            y_out = np.dot(y_hid_sigmoid, weight_hid_to_out) + bias_out
            y_out_softmax = self.__softmax(y_out)

            # start backpropagation
            error = y_out_softmax - y
            error_weight_hid_to_out = np.dot(y_hid_sigmoid.T, error)
            error_bias_out = error

            error_hid = np.dot(error, weight_hid_to_out.T)
            f_deriv = self.__sigmoid_prime(y_hid)
            error_bias_hid = error_hid * f_deriv
            error_weight_in_to_hid = np.dot(x.T, error_bias_hid)

            # update weights and bias
            weight_in_to_hid -= self.__learning_rate * error_weight_in_to_hid
            bias_hid -= self.__learning_rate * error_bias_hid.sum(axis=0)
            weight_hid_to_out -= self.__learning_rate * error_weight_hid_to_out
            bias_out -= self.__learning_rate * error_bias_out.sum(axis=0)

        self.__weights = [weight_in_to_hid, weight_hid_to_out]
        self.__bias = [bias_hid, bias_out]

    def predict(self, x):
        weight_in_to_hid, weight_hid_to_out = self.__weights
        bias_hid, bias_out = self.__bias
        y_hid = np.dot(x, weight_in_to_hid) + bias_hid
        y_hid_sigmoid = self.__sigmoid(y_hid)
        y_out = np.dot(y_hid_sigmoid, weight_hid_to_out) + bias_out
        y_out_softmax = self.__softmax(y_out)
        computed_output = [list(output).index(max(output)) for output in y_out_softmax]
        return computed_output
