import numpy as np
from sklearn import linear_model
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


def split_data_multitarget(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(outputs[0]))]
    train_sample = np.random.choice(indexes, int(0.8 * len(outputs[0])), replace=False)
    validation_sample = [i for i in indexes if i not in train_sample]
    train_inputs = []
    validation_inputs = []
    for inp in inputs:
        train_inputs.append([inp[i] for i in train_sample])
        validation_inputs.append([inp[i] for i in validation_sample])
    train_outputs = []
    validation_outputs = []
    for out in outputs:
        train_outputs.append([out[i] for i in train_sample])
        validation_outputs.append([out[i] for i in validation_sample])
    return train_inputs, train_outputs, validation_inputs, validation_outputs


def multitarget_regression():
    input_multitarget, output_multitarget = \
        make_regression(n_samples=1000, n_features=2, n_targets=2, random_state=1, noise=0.5)
    np.random.seed(10)
    train_indexes = np.random.choice([i for i in range(len(input_multitarget))],
                                     int(0.8 * len(input_multitarget)), replace=False)
    validation_indexes = [i for i in range(len(input_multitarget)) if i not in train_indexes]
    train_inputs = [input_multitarget[i] for i in train_indexes]
    train_outputs = [output_multitarget[i] for i in train_indexes]
    validation_inputs = [input_multitarget[i] for i in validation_indexes]
    validation_outputs = [output_multitarget[i] for i in validation_indexes]
    model = linear_model.LinearRegression()
    model.fit(train_inputs, train_outputs)
    plt.plot(*list(map(list, zip(*validation_outputs))), "ro")
    plt.plot(*list(map(list, zip(*model.predict(validation_inputs)))), "g*")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Multitarget")
    plt.show()
    for i in range(len(model.intercept_)):
        print(i, " My learnt model: f(x) = ", model.intercept_[i], "+", model.coef_[i][0], "* x1", "+", model.coef_[i][1]
              , "* x2")
    return model.intercept_, model.coef_, model.predict(validation_inputs), validation_outputs