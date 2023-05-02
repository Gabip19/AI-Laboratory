import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("TkAgg")


def read_from_csv(filename, inputs, output):
    file = pd.read_csv(filename)
    features = []
    for i in inputs:
        features.append([float(value) for value in file[i]])
    outputs = [float(value) for value in file[output]]
    return features, outputs


def read_from_csv_multitarget(filename, inputs, output):
    file = pd.read_csv(filename)
    features = []
    outputs = []
    for i in inputs:
        features.append([float(value) for value in file[i]])
    for o in output:
        outputs.append([float(value) for value in file[o]])
    return features, outputs


def prepare_data(features, result):
    matrix = []
    for index, elems in enumerate(zip(features[0], features[1])):
        first, second = elems
        if pd.isna(first) or pd.isna(second) or first == second == 0 or [first, second] in matrix:
            features[0].pop(index)
            features[1].pop(index)

            if len(result) == 2:
                result[0].pop(index)
                result[1].pop(index)
            else:
                result.pop(index)
        else:
            matrix.append([first, second])
    return features, result


def plotData(x1, y1, x2=None, y2=None, x3=None, y3=None, title=None):
    plt.plot(x1, y1, 'ro', label='train data')
    if x2:
        plt.plot(x2, y2, 'b-', label='learnt model')
    if x3:
        plt.plot(x3, y3, 'g^', label='test data')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_data_linearity(train_inputs, train_output, title=None, show_plot=True, ax=None):
    ax = plt.axes(projection='3d') if ax is None else ax

    ax.set_xlabel("capita")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")

    ax.plot3D(train_inputs[0], train_inputs[1], train_output, 'ro')
    ax.set_title(title)

    if show_plot:
        plt.show()

    return ax


def plot_results(train_inputs, train_output, w0, w1, w2):
    min1 = min(train_inputs[0])
    max1 = max(train_inputs[0])

    min2 = min(train_inputs[1])
    max2 = max(train_inputs[1])

    ax = plt.axes(projection='3d')
    xref1, xref2 = np.meshgrid(np.arange(min1, max1 + 0.1, max1 - min1),
                               np.arange(min2, max2 + 0.1, max2 - min2))
    yref = xref2 * w2 + xref1 * w1 + w0
    ax.plot_surface(xref1, xref2, yref, alpha=0.5, label='learnt model')
    plot_data_linearity(train_inputs, train_output, 'plot', False, ax)
    plt.show()


def plot3Ddata(x1Train, x2Train, yTrain, x1Model=None, x2Model=None, yModel=None, x1Test=None, x2Test=None, yTest=None,
               title=None):
    ax = plt.axes(projection='3d')
    if x1Train:
        plt.scatter(x1Train, x2Train, yTrain, c='r', marker='o', label='train data')
    if x1Model:
        plt.scatter(x1Model, x2Model, yModel, c='b', marker='_', label='learnt model')
    if x1Test:
        plt.scatter(x1Test, x2Test, yTest, c='g', marker='^', label='test data')
    plt.title(title)
    ax.set_xlabel("capita")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.legend()
    plt.show()
