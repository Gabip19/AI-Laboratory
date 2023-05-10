import random
from math import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


class MyMulticlassLogisticRegression:
    def __init__(self):
        self.intercept_ = []
        self.coef_ = []

    def fit(self, x, y, learning_rate=0.001, no_epochs=2000):
        self.intercept_ = []
        self.coef_ = []
        labels = list(set(y))
        for label in labels:
            coefficient = [random.random() for _ in range(len(x[0]) + 1)]
            for epoch in range(no_epochs):
                errors = [0] * len(coefficient)
                for input_, output_ in zip(x, y):
                    y_computed = sigmoid(self.eval(input_, coefficient))
                    error = y_computed - 1 if output_ == label else y_computed
                    for i, xi in enumerate([1] + list(input_)):
                        errors[i] += error * xi
                for i in range(len(coefficient)):
                    coefficient[i] = coefficient[i] - learning_rate * errors[i]
            self.intercept_.append(coefficient[0])
            self.coef_.append(coefficient[1:])

    @staticmethod
    def eval(xi, coefficients):
        yi = coefficients[0]
        for j in range(len(xi)):
            yi += coefficients[j + 1] * xi[j]
        return yi

    def predict_one_sample(self, sample_features):
        predictions = []
        for intercept, coefficient in zip(self.intercept_, self.coef_):
            computed_value = self.eval(sample_features, [intercept] + coefficient)
            predictions.append(sigmoid(computed_value))
        return predictions.index(max(predictions))

    def predict(self, samples):
        computed_labels = [self.predict_one_sample(sample) for sample in samples]
        return computed_labels
