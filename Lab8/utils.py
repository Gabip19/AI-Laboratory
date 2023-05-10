import plotly.express as px
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def load_data_flowers():
    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    outputs_name = data['target_names']
    feature_names = list(data['feature_names'])
    feature1 = [feat[feature_names.index('sepal length (cm)')] for feat in inputs]
    feature2 = [feat[feature_names.index('sepal width (cm)')] for feat in inputs]
    feature3 = [feat[feature_names.index('petal length (cm)')] for feat in inputs]
    feature4 = [feat[feature_names.index('petal width (cm)')] for feat in inputs]
    inputs = [[feat[feature_names.index('sepal length (cm)')],
               feat[feature_names.index('sepal width (cm)')],
               feat[feature_names.index('petal length (cm)')],
               feat[feature_names.index('petal width (cm)')]] for feat in inputs]
    return inputs, outputs, outputs_name, feature1, feature2, feature3, feature4, feature_names


def plot_histogram_feature(feature, variableName):
    plt.hist(feature, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def print_regression_models(regressor):
    w0, w1, w2, w3, w4 = regressor.intercept_[0], regressor.coef_[0][0], regressor.coef_[0][1], \
        regressor.coef_[0][2], regressor.coef_[0][3]
    print("First label: y =", w0, '+', w1, "* x1 +", w2, "* x2 +", w3, "* x3 +", w4, "* x4")

    w0, w1, w2, w3, w4 = regressor.intercept_[1], regressor.coef_[1][0], regressor.coef_[1][1], \
        regressor.coef_[1][2], regressor.coef_[1][3]
    print("Second label: y =", w0, '+', w1, "* x1 +", w2, "* x2 +", w3, "* x3 +", w4, "* x4")

    w0, w1, w2, w3, w4 = regressor.intercept_[2], regressor.coef_[2][0], regressor.coef_[2][1], \
        regressor.coef_[2][2], regressor.coef_[2][3]
    print("Third label: y =", w0, '+', w1, "* x1 +", w2, "* x2 +", w3, "* x3 +", w4, "* x4")


def plot_predictions(inputs, real_outputs, computed_outputs, label_names, feature_names, title=None):
    labels = list(set(outputs))
    no_data = len(inputs)
    for crt_label in labels:
        x = [inputs[i][0] for i in range(no_data) if real_outputs[i] == crt_label and computed_outputs[i] == crt_label]
        y = [inputs[i][1] for i in range(no_data) if real_outputs[i] == crt_label and computed_outputs[i] == crt_label]
        plt.scatter(x, y, label=label_names[crt_label] + ' (correct)')
    for crt_label in labels:
        x = [inputs[i][0] for i in range(no_data) if real_outputs[i] == crt_label and computed_outputs[i] != crt_label]
        y = [inputs[i][1] for i in range(no_data) if real_outputs[i] == crt_label and computed_outputs[i] != crt_label]
        plt.scatter(x, y, label=label_names[crt_label] + ' (incorrect)')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.legend()
    plt.show()


# TODO: modifica asta
def plot_data_four_features(inputs, outputs, output_names, feature_names, title=None):
    x = [i[0] for i in inputs]
    y = [i[1] for i in inputs]
    z = [i[2] for i in inputs]
    v = [i[3] for i in inputs]
    figure = px.scatter_3d(x=x, y=y, z=z, symbol=v, color=outputs, title=title,
                           labels=dict(x=feature_names[0], y=feature_names[1], z=feature_names[2],
                                       symbol=feature_names[3], color="Type"))
    figure.update_layout(legend=dict(orientation="v", yanchor='top', xanchor="right"))
    figure.show()