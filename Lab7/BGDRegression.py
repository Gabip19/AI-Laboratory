import random
from statistics import mean


class MyBGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    def fit(self, x, y, learning_rate=0.001, no_epochs=1000):
        self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]
        for epoch in range(no_epochs):
            errors = []
            for i in range(len(x)):
                y_computed = self.eval(x[i])
                errors.append(y_computed - y[i])
            error = mean(errors)
            for i in range(len(x)):
                for j in range(len(x[0])):
                    self.coef_[j] = self.coef_[j] - learning_rate * error * x[i][j]
                self.coef_[len(x[0])] = self.coef_[len(x[0])] - learning_rate * error * 1

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi

    def predict(self, x):
        return [self.intercept_ + sum([self.coef_[i] * e[i] for i in range(len(e))]) for e in x]
